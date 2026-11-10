"""cond-pack：把 COND 打包成结构化条件（复杂多条件编排的起点）。

对标 ComfyUI 的 cond 包装链，一条裸 cond 依次可加：
- weight：整体权重
- [start_percent, end_percent)：生效的采样进度窗口
- [layer_start, layer_end)：生效的 cross-attention 层区间（浅层管构图、深层管细节）

输出可继续用 cond-concat 并列，或接 cond-set-area 做区域限定。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()
    ensure_plugin(url, "cond.pack", tok=tok)

    cond_id = param("cond", "")
    if not cond_id:
        raise RuntimeError("cond-pack: cond 必填")

    inputs = {
        "cond": ref(cond_id),
        "weight": param("weight", 1.0, float),
        "start_percent": param("start_percent", 0.0, float),
        "end_percent": param("end_percent", 1.0, float),
        "layer_start": param("layer_start", 0, int),
        "layer_end": param("layer_end", -1, int),
    }
    jid = submit_job(url, {"name": "cond.pack", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[cond-pack] weight={inputs['weight']} "
          f"window=[{inputs['start_percent']},{inputs['end_percent']}) "
          f"layers=[{inputs['layer_start']},{inputs['layer_end']})", flush=True)
    result = wait_job(url, jid, tok, timeout=param("job_timeout", 300, int),
                      poll=param("poll_interval", 2, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[cond-pack] -> cond={out['cond']}", flush=True)
    emit(cond=out["cond"])


if __name__ == "__main__":
    main()
