"""inference-op：通用算子调用节点。新算子无需新增节点包。

inputs_json 中对象引用写法：{"$id": "uuid"}；
workflow 接线时用参数绑定填充，如：
  inputs_json: '{"clip": {"$id": "{{ LoadCheckpoint.clip_ref }}"}, "text": "a cat"}'

执行方式：统一走异步 POST /jobs + 轮询（wait_job）。
- inputs.preview_every>0 时逐轮询上报 preview 帧地址（FLOWX_PREVIEW 标记），
  画布实时显示采样过程帧 + 进度；
- 完成后对 IMAGE 类型输出上报完成帧（{base}/images/{id}），结果图直接上画布；
- emit 附带内部元数据 __op_name / __inputs_resolved（解析后的真实入参），
  供 Studio op-replay 端点重放（预处理算子调参预览）；widget 显示时过滤 __ 前缀。
"""
import json

from flowx_client import (emit_preview, param, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    op_name = param("op_name")
    inputs_json = param("inputs_json", "{}")
    emit_keys = [k.strip() for k in param("emit_keys", "").split(",") if k.strip()]
    tok = token()

    try:
        inputs = json.loads(inputs_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"inputs_json is not valid JSON: {e}")

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

    result = wait_job(url, jid, tok, timeout=3600, poll=2.0, on_poll=on_poll)
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


if __name__ == "__main__":
    main()
