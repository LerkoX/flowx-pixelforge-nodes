"""pose-gate：姿态检测 + 人工确认闸门节点。

流程：pose.detect（服务端 DWPose，返回骨架图 + 关键点 JSON）→ 创建
/ext/gate/{execution}:{node} 会话并上报骨架预览帧 → 轮询等待 widget 决策：
  - confirm（原骨架无误）→ 直接用检测骨架继续；
  - render（widget 拖拽修正了关键点）→ 服务端已按新关键点重渲染，用新骨架；
  - 超时 → on_timeout=pass 用原骨架继续（无人值守等价旧行为）/ fail 报错。
会话 TTL 1 小时，正常结束/超时后节点主动 DELETE 清理。
"""
import json
import os
import time

from flowx_client import (call_op, delete_json, emit, emit_result_preview,
                          ensure_plugin, flush_print, get_bytes, get_json,
                          param, post_bytes, post_json, ref, token)

POLL_INTERVAL = 3.0


def _bool(v, default=False):
    if v is None or str(v).strip() == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def main():
    base = param("service_url").rstrip("/")
    tok = token()
    image = param("image")
    include_hand = _bool(param("include_hand", "true"), True)
    include_face = _bool(param("include_face", "false"), False)
    timeout_sec = int(float(param("timeout_sec", "600")))
    on_timeout = (param("on_timeout", "pass") or "pass").strip().lower()

    # 算子 + 路由自注册（内容 hash 幂等）
    op = ensure_plugin(base, "pose.detect", tok=tok)

    flush_print(f"[pose-gate] 检测中（DWPose, hand={include_hand} "
                f"face={include_face}）…")
    res = call_op(base, op, {"image": ref(image),
                             "include_hand": include_hand,
                             "include_face": include_face}, tok=tok)
    skel_id = res["image"]
    kp = json.loads(res["keypoints"])
    flush_print(f"[pose-gate] 检测完成：骨架 {skel_id} 画布 "
                f"{kp['width']}x{kp['height']}")

    key = f"{os.environ.get('FLOWX_EXECUTION_ID', '0')}:" \
          f"{os.environ.get('FLOWX_NODE_ID', 'node')}"
    post_json(base, f"/ext/gate/{key}",
              {"image_id": skel_id, "source_id": image, "keypoints": kp},
              tok, timeout=30)
    # 骨架图先上画布（widget 编辑器也经 service-proxy 自取会话/原图）
    emit_result_preview(base, skel_id, tok)
    flush_print(f"[pose-gate] 等待人工确认（gate={key}，超时 {timeout_sec}s，"
                f"超时策略 {on_timeout}）：画布节点上拖拽修正关键点后点"
                f"「确认放行」，或「原样继续」")

    deadline = time.time() + timeout_sec
    action, final_id = "timeout", skel_id
    try:
        while time.time() < deadline:
            time.sleep(POLL_INTERVAL)
            try:
                s = get_json(base, f"/ext/gate/{key}", tok, timeout=15)
            except Exception as e:  # 瞬态抖动/会话短暂不可见不致命
                flush_print(f"[pose-gate] 轮询异常（忽略继续等）：{e}")
                continue
            dec = s.get("decision")
            if dec:
                action, final_id = dec["action"], dec["image_id"]
                break

        if action == "timeout":
            if on_timeout == "fail":
                raise SystemExit(
                    f"pose-gate: 等待人工确认超时（{timeout_sec}s），"
                    f"on_timeout=fail，节点中止")
            flush_print(f"[pose-gate] 超时未确认，按 on_timeout=pass 用原骨架继续")
            action = "timeout_pass"
        elif action == "render":
            # 修正骨架在本会话里是 PNG 字节：拉回来经 POST /images 入对象库。
            # ⚠️ 必须在 DELETE 会话之前取（下面 finally 会清掉会话）
            png = get_bytes(base, f"/ext/gate/{key}/image", tok, timeout=60)
            final_id = post_bytes(base, "/images", png, tok, timeout=120,
                                  content_type="image/png")["id"]
            flush_print(f"[pose-gate] 修正骨架已入库：{final_id}")
        else:
            flush_print(f"[pose-gate] 收到人工决策：{action} -> {final_id}")
    finally:
        try:
            delete_json(base, f"/ext/gate/{key}", tok, timeout=10)
        except Exception:
            pass

    if final_id != skel_id:
        emit_result_preview(base, final_id, tok)  # 修正后的骨架上画布
    emit(image=final_id, gate_action=action,
         keypoints_detected="1")


if __name__ == "__main__":
    main()
