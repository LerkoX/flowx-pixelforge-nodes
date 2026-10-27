"""vae-encode-inpaint-masked：原生 9ch inpaint 的 masked_image_latents 编码。

关键语义：先在线图像素空间把白色 mask 区域置黑，再做 VAE encode。
不能先 VAE encode 完整图像再乘 latent mask——VAE encode 与空间 mask 不可交换。
任务式提交（/jobs + 轮询），避免隧道在 GPU 队列等待期间掐断同步请求。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()
    ensure_plugin(url, "sd.vae.encode_inpaint_masked", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {
        "name": "sd.vae.encode_inpaint_masked",
        "inputs": {
            "vae": ref(param("vae_ref")),
            "image": ref(param("image")),
            "mask": ref(param("mask")),
        },
    }, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[encode-inpaint-masked] image={param('image')} mask={param('mask')} -> "
          f"latent={out['latent']} in {time.time()-t0:.1f}s", flush=True)
    emit(latent=out["latent"])


if __name__ == "__main__":
    main()
