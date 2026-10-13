"""vram-ops：显存/常驻/租约/策略运维节点（单流水线串行时的实验工具）。

为什么需要（dev-log §58.9 实测）：失败或中断的执行会留下 **30 分钟租约**，租约持有的
常驻条目不会被驱逐 ⇒ 下一次冷加载等 300s 后失败（`UNet+VAE+reserve > 8G`）。当时只能
靠人肉 curl；本节点把"看一眼/清一下/改旋钮"变成流水线里可复用的一个节点。

动作（`action`）：
- `status`      ：`GET /health` 摘要（常驻条目数/显存余量/淘汰开销）
- `leases`      ：`GET /leases`（活跃租约及持有的常驻条目 —— 排查"排队不轮到"第一现场）
- `unload`      ：`POST /model/unload`（`target` 空 = 全部；带 `lease` 时按该租约视角清）
- `interrupt`   ：`POST /interrupt`（`job_id` 空 = 取消全部 running/pending）
- `policy_get`  ：`GET /model/policy`（读显存淘汰策略旋钮）
- `policy_set`  ：`POST /model/policy`（写 `policy_json`，如 {"vram_reserve_mb": 768}）

输出（**扁平**，值里不带 `#`/代码围栏 —— Studio codec 会截断）：`ok`/`action`/`detail`/
`resident_count`/`lease_count`/`vram_free_mb`/`policy`（"k=v,k=v" 形式）。
"""
import json

from flowx_client import emit, get_json, param, post_json, token

ACTIONS = ("status", "leases", "unload", "interrupt", "policy_get", "policy_set")


def _flat(d, keys):
    """取若干字段拼成 "k=v,k=v"（**不带花括号**：值必须能安全落进 YAML 代码块）。"""
    parts = []
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] is not None:
            parts.append(f"{k}={d[k]}")
    return ",".join(parts)


def _explain(e):
    """把"端点不存在"翻译成人话：策略端点的代码已就绪但需服务端重新部署。"""
    msg = str(e)
    if "HTTP 404" in msg or "HTTP 405" in msg:
        return RuntimeError(
            "vram-ops: 服务端未提供该端点（" + msg.split("->")[-1].strip() +
            "）。/model/policy 的实现在 inference-server/app/policy.py + main.py 已就绪，"
            "但需**重新部署推理服务**后生效；status/leases/unload/interrupt 不受影响。")
    return e


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    action = param("action", "status")
    if action not in ACTIONS:
        raise ValueError(f"vram-ops: action 非法 {action!r}（可选 {list(ACTIONS)}）")
    lease = param("lease", "")
    headers = {"X-FlowX-Lease": lease} if lease else None

    try:
        resident_count = lease_count = vram_free_mb = 0
        policy_txt = ""

        if action == "status":
            h = get_json(url, "/health", tok, headers=headers)
            resident_count = len(h.get("resident_models") or [])
            vram_free_mb = int(h.get("vram_free_mb") or 0)
            detail = ("gpu=%s vram_free=%sMB resident=%s" % (
                h.get("gpu_name"), vram_free_mb, h.get("resident_models")))
        elif action == "leases":
            lz = get_json(url, "/leases", tok, headers=headers)
            leases = lz.get("leases") or []
            lease_count = len(leases)
            detail = "; ".join(
                "%s→%s(idle%.0fs)" % (l["lease"], ",".join(l.get("keys") or []) or "-",
                                      float(l.get("idle_seconds") or 0))
                for l in leases) or "无活跃租约"
            detail += " | ttl=%ss wait=%ss" % (lz.get("lease_ttl_seconds"),
                                               lz.get("lease_wait_timeout_seconds"))
        elif action == "unload":
            r = post_json(url, "/model/unload", {"target": param("target", "")}, tok,
                          headers=headers)
            detail = "unloaded=%s resident=%s" % (r.get("unloaded"), r.get("resident"))
            resident_count = len(str(r.get("resident") or "").split(",")) if r.get("resident") else 0
        elif action == "interrupt":
            jid = param("job_id", "")
            r = post_json(url, "/interrupt", ({"job_id": jid} if jid else {}), tok,
                          headers=headers)
            detail = "interrupted=%s（%s）" % (r.get("interrupted"),
                                              "指定 job" if jid else "全部 running/pending")
        elif action == "policy_get":
            p = get_json(url, "/model/policy", tok, headers=headers)
            policy_txt = _flat(p, ("max_resident_models", "vram_reserve_mb",
                                   "lease_ttl_seconds", "lease_wait_timeout_seconds",
                                   "offload_mode", "quantization"))
            detail = policy_txt
        else:  # policy_set
            raw = param("policy_json", "{}")
            try:
                patch = json.loads(raw)
            except ValueError as e:
                raise ValueError(f"vram-ops: policy_json 不是合法 JSON（{e}）：{raw!r}")
            r = post_json(url, "/model/policy", patch, tok, headers=headers)
            applied = r.get("applied") or {}
            policy_txt = _flat(r.get("policy") or {}, ("max_resident_models",
                                                       "vram_reserve_mb"))
            detail = "applied=%s policy=%s" % (
                ",".join(f"{k}={v}" for k, v in applied.items()), policy_txt)

    except RuntimeError as e:
        raise _explain(e)

    print(f"[vram-ops] {action}: {detail}", flush=True)
    emit(ok="true", action=action, detail=str(detail)[:500],
         resident_count=resident_count, lease_count=lease_count,
         vram_free_mb=vram_free_mb, policy=policy_txt)


if __name__ == "__main__":
    main()
