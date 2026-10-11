"""flux-mask-refine：FLUX 蒙版局部重绘节点

自包含节点：服务端算子 flux.mask_refine 由 server_op.py 自注册 (ensure_plugin)。
消费 flux-unet-load 的 MODEL + flux-vae-load 的 VAE + flux-redux-apply 的 COND
+ 模特原图 IMAGE + 蒙版 MASK，输出局部重绘后无缝贴回的完整图像。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "flux.mask_refine", tok=tok)

    inputs = {
        "model": ref(param("model")),
        "vae": ref(param("vae")),
        "cond": ref(param("cond")),
        "image": ref(param("image")),
        "mask": ref(param("mask")),
        "padding": param("padding", 0.3, float),
        "denoise": param("denoise", 0.8, float),
        "steps": param("steps", 8, int),
        "guidance": param("guidance", 0.0, float),
        "seed": param("seed", -1, int),
        "guide_size": param("guide_size", 768, int),
        "feather": param("feather", 16, int),
    }
    print(f"[flux-mask-refine] denoise={inputs['denoise']} steps={inputs['steps']} "
          f"image={param('image')} mask={param('mask')}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.mask_refine", "inputs": inputs}, tok)

    def on_poll(view):
        p = view.get("progress") or {}
        cur, tot = p.get("current", 0), p.get("total", 0)
        if cur > 0 and tot > 0:
            emit_preview(f"{pbase}/preview/{jid}", cur / tot, tok,
                         base=pbase, job_id=jid)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 14400, int),
                      poll=param("poll_interval", 5, int),
                      on_poll=on_poll)
    outs = result["outputs"]
    image_id = outs["image"]["id"]
    print(f"[flux-mask-refine] done in {time.time()-t0:.1f}s -> image={image_id}", flush=True)
    emit(image=image_id)


if __name__ == "__main__":
    main()
