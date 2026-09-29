"""flux-sampler：FLUX 采样器（对标 ComfyUI SamplerCustomAdvanced）。

自包含节点：服务端算子 flux.sample 由 server_op.py 自注册（ensure_plugin）。
消费 flux-unet-load 的 MODEL + flux-clip-encode 的 COND +
flux-empty-latent/flux-vae-encode 的 LATENT，输出采样后 LATENT
（VAE 解码独立成 flux-vae-decode 节点）。
latent 来自 empty-latent → txt2img；来自 vae-encode 且 denoise<1 → img2img
（hires-fix 两阶段可编排）。schnell 为 1~4 步蒸馏模型：steps 默认 4、
guidance 固定 0（免负向）。异步 job + 逐步 latent 预览帧。
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
    seed = param("seed", -1, int)
    denoise = param("denoise", 1.0, float)
    inputs = {
        "model": ref(param("model")),
        "cond": ref(param("cond")),
        "latent": ref(param("latent")),
        "seed": seed,
        "steps": steps,
        "guidance": param("guidance", 0.0, float),
        "denoise": denoise,
        "preview_every": preview_every,
    }
    print(f"[flux-sampler] steps={steps} denoise={denoise} seed={seed}",
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
    latent_id = outs["latent"]["id"]
    actual_seed = outs["seed"].get("value")
    print(f"[flux-sampler] done in {time.time()-t0:.1f}s -> latent={latent_id} "
          f"seed={actual_seed}", flush=True)
    emit(latent=latent_id, seed=actual_seed)


if __name__ == "__main__":
    main()
