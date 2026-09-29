"""flux-detail-refine：FLUX ADetailer 式局部重绘（脸/手检测 → 裁剪放大 →
FLUX img2img 低 strength 重绘 → 羽化贴回）。

自包含节点：服务端算子 flux.detail_refine 由 server_op.py 自注册
（ensure_plugin）。消费 flux-unet-load 的 MODEL + flux-vae-load 的 VAE +
flux-clip-encode 的 COND + 上游图像，输出重绘后图像。
异步 job + 进度/合成预览帧（检测框调试帧 + 逐目标贴回帧）。
修脸 denoise 0.35~0.5，修手 0.45~0.6；steps 默认 8（实际去噪步数
≈ steps×denoise，schnell 蒸馏勿用 4 步配低 denoise）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)  # 画布预览帧由 Studio 取，必须宿主机可达
    tok = token()
    detector = param("detector", "face")

    ensure_plugin(url, "flux.detail_refine", tok=tok)

    inputs = {
        "model": ref(param("model")),
        "vae": ref(param("vae")),
        "cond": ref(param("cond")),
        "image": ref(param("image")),
        "detector": detector,
        "conf": param("conf", 0.3, float),
        "padding": param("padding", 0.4, float),
        "denoise": param("denoise", 0.4, float),
        "steps": param("steps", 8, int),
        "guidance": param("guidance", 0.0, float),
        "seed": param("seed", -1, int),
        "guide_size": param("guide_size", 512, int),
        "max_targets": param("max_targets", 4, int),
        "feather": param("feather", 16, int),
    }
    print(f"[flux-detail-refine] detector={detector} "
          f"denoise={inputs['denoise']} steps={inputs['steps']} "
          f"image={param('image')}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.detail_refine", "inputs": inputs}, tok)

    def on_poll(view):
        p = view.get("progress") or {}
        cur, tot = p.get("current", 0), p.get("total", 0)
        if cur > 0 and tot > 0:
            emit_preview(f"{pbase}/preview/{jid}", cur / tot, tok,
                         base=pbase, job_id=jid)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int),
                      on_poll=on_poll)
    outs = result["outputs"]
    image_id = outs["image"]["id"]
    count = outs["count"].get("value")
    print(f"[flux-detail-refine] done in {time.time()-t0:.1f}s "
          f"refined={count} -> image={image_id}", flush=True)
    emit(image=image_id, count=str(count))


if __name__ == "__main__":
    main()
