"""qwen-image-vae-load：Qwen-Image VAE 加载（AutoencoderKLQwenImage fp32）。"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)



def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "qwen_image.vae_load", tok=tok)

    inputs = {
        "model": param("model", "Qwen-Image-2512"),
        "vae": param("vae", ""),
        "device": param("device", "cuda"),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "qwen_image.vae_load", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[qwen-image-vae-load] job={jid} inputs={inputs}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[qwen-image-vae-load] -> vae={out['vae']} ({time.time()-t0:.1f}s)", flush=True)
    emit(vae=out['vae'])


if __name__ == "__main__":
    main()
