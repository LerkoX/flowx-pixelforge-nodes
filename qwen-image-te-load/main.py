"""qwen-image-te-load：Qwen-Image 文本编码器加载（Qwen2.5-VL-7B bf16 + tokenizer）。"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)



def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "qwen_image.te_load", tok=tok)

    inputs = {
        "model": param("model", "Qwen-Image-2512"),
        "dtype": param("dtype", "bf16"),
        "offload": param("offload", "sequential"),
        "group_blocks": param("group_blocks", 2, int),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "qwen_image.te_load", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[qwen-image-te-load] job={jid} inputs={inputs}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[qwen-image-te-load] -> text_encoder={out['text_encoder']} ({time.time()-t0:.1f}s)", flush=True)
    emit(text_encoder=out['text_encoder'])


if __name__ == "__main__":
    main()
