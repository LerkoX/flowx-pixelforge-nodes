"""flux-sampler：FLUX.1-schnell 采样器（distilled guidance，免负向）。

自包含节点：服务端算子 flux.sample 由 server_op.py 自注册（ensure_plugin）。
消费 flux-loader 的 model 引用 + flux-clip-encode 的 cond，直接输出图像
（VAE 解码内置，无需独立 decode 节点）。异步 job + 进度卡片预览。
schnell 为 1~4 步蒸馏模型：steps 默认 4、guidance 固定 0（不填负向）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)  # 画布预览帧由 Studio 取，必须宿主机可达
    tok = token()

    ensure_plugin(url, "flux.sample", tok=tok)

    preview_every = param("preview_every", 1, int)
    steps = param("steps", 4, int)
    width = param("width", 1024, int)
    height = param("height", 1024, int)
    seed = param("seed", -1, int)
    inputs = {
        "model": ref(param("model_ref")),
        "cond": ref(param("cond")),
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "guidance": param("guidance", 0.0, float),
        "preview_every": preview_every,
    }
    print(f"[flux-sampler] {width}x{height} steps={steps} seed={seed}",
          flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.sample", "inputs": inputs}, tok)

    def on_poll(view):
        if preview_every <= 0:
            return
        p = view.get("progress") or {}
        cur, tot = p.get("current", 0), p.get("total", 0)
        if cur > 0 and tot > 0:
            emit_preview(f"{pbase}/preview/{jid}", cur / tot, tok,
                         base=pbase, job_id=jid)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 7200, int),
                      poll=param("poll_interval", 5, int),
                      on_poll=on_poll)
    outs = result["outputs"]
    image_id = outs["image"]["id"]
    actual_seed = outs["seed"].get("value")
    print(f"[flux-sampler] done in {time.time()-t0:.1f}s -> image={image_id} "
          f"seed={actual_seed}", flush=True)
    emit(image=image_id, seed=actual_seed)


if __name__ == "__main__":
    main()
