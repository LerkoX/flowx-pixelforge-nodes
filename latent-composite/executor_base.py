"""executor_base：专属节点共享执行器（异步 job + 预览上报 + 重放元数据）。

各专属节点 main.py 只需声明算子与参数映射，~20 行：

    from executor_base import run_op
    from flowx_client import param, ref

    def main():
        run_op("controlnet.apply", {
            "control_net": ref(param("control_net")),
            "image": ref(param("image")),
            "strength": param("strength", 1.0, cast=float),
        }, emit_keys=["control"])

    if __name__ == "__main__":
        main()

行为（与 inference-op 第一批增强一致，单一事实来源）：
- POST /jobs 异步提交 + wait_job 轮询（2s），超时 wait_job 自动 interrupt 防孤儿；
- inputs.preview_every>0 时逐轮询上报 preview 帧（FLOWX_PREVIEW 标记，
  画布实时显示采样过程帧 + 进度）；
- 完成后对 IMAGE 类型输出上报完成帧（{base}/images/{id}），
  无采样过程的算子（预处理/解码/放大）结果图也直接上画布；
- 输出经 ```flowx-yaml 块：outputs_json + __op_name + __inputs_resolved
  + emit_keys 指定键；__ 前缀内部元数据供 Studio op-replay 端点重放
  （预处理调参即时预览），widget 显示时过滤 __ 前缀；
- check_exists=True 时先查 /ops 校验算子已注册（插件算子缺失给出明确报错）。
"""
import json

from flowx_client import (emit_preview, get_json, param, submit_job, token,
                          wait_job)


def check_op_exists(url, tok, op_name):
    """插件算子存在性校验：算子实现以服务端（plugins/）为唯一来源，
    节点包不自注册实现；缺失时给出可操作的报错。"""
    ops = get_json(url, "/ops", tok, timeout=30).get("ops", [])
    if not any(o.get("name") == op_name for o in ops):
        raise RuntimeError(
            f"算子 '{op_name}' 未注册：推理服务侧插件未部署"
            f"（该算子实现以服务端为唯一来源，节点包不自注册）")


def run_op(op_name, inputs, emit_keys=(), check_exists=False, timeout=3600):
    """提交单个算子为异步 job 并回传输出。inputs 中对象引用用 ref() 包装；
    service_url/service_token 从节点参数读（flowx.json env 映射）。"""
    url = param("service_url").rstrip("/")
    tok = token()

    if check_exists:
        check_op_exists(url, tok, op_name)

    jid = submit_job(url, {"name": op_name, "inputs": inputs}, tok)
    print(f"[op] {op_name} job={jid[:8]}", flush=True)

    want_preview = False
    try:
        want_preview = int(inputs.get("preview_every", 0) or 0) > 0
    except (TypeError, ValueError):
        pass

    def on_poll(v):
        if not want_preview:
            return
        p = v.get("progress") or {}
        cur, tot = p.get("current", 0), p.get("total", 0)
        prog = (cur / tot) if tot else None
        emit_preview(f"{url}/preview/{jid}", prog, tok, base=url, job_id=jid)

    result = wait_job(url, jid, tok, timeout=timeout, poll=2.0, on_poll=on_poll)
    outputs = (result or {}).get("outputs", {})
    flat = {k: m.get("id", m.get("value")) for k, m in outputs.items()}
    print(f"[op] {op_name} -> {json.dumps(flat, ensure_ascii=False)}", flush=True)

    # 完成帧兜底：IMAGE 输出上报最终图（无采样过程的算子也能出结果图）
    for k, m in outputs.items():
        if m.get("type") == "IMAGE" and m.get("id"):
            emit_preview(f"{url}/images/{m['id']}", 1.0, tok,
                         base=url, job_id=jid)

    print("```flowx-yaml")
    print(f"outputs_json: {json.dumps(flat, ensure_ascii=False)}")
    # 内部元数据（__ 前缀）：op-replay 重放用；widget 显示时过滤
    print(f'__op_name: "{op_name}"')
    print(f"__inputs_resolved: {json.dumps(inputs, ensure_ascii=False)}")
    for k in emit_keys:
        if k in flat:
            print(f'{k}: "{flat[k]}"')
    print("```")
    return flat
