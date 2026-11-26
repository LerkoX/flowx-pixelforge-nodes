"""qwen-image-transformer-load：Qwen-Image transformer 组件加载（20B，逐层 offload）。"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)



def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "qwen_image.transformer_load", tok=tok)

    inputs = {
        "model": param("model", "Qwen-Image-2512"),
        "transformer": param("transformer", ""),
        "dtype": param("dtype", "bf16"),
        "offload": param("offload", "sequential"),
        "group_blocks": param("group_blocks", 4, int),
        "quant": param("quant", ""),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "qwen_image.transformer_load", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[qwen-image-transformer-load] job={jid} inputs={inputs}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[qwen-image-transformer-load] -> transformer={out['transformer']} ({time.time()-t0:.1f}s)", flush=True)
    emit(transformer=out['transformer'])


if __name__ == "__main__":
    main()
