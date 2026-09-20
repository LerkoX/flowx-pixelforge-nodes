"""sd3-sampler：SD3.5 采样器（flow matching），消费模型引用 + 正/反 COND + latent。

自包含节点：服务端算子 sd3.sample 由 server_op.py 自注册（ensure_plugin）。
异步任务通道（POST /jobs + 轮询）；preview_every>0 时进度卡片帧地址经
FLOWX_PREVIEW 标记上报 Studio 中转拉帧（与 ksampler 同模式）。
denoise=1 文生图；denoise<1 接 sd3-vae-encode 的 latent 做图生图。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, param, ref,
                          submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "sd3.sample", tok=tok)

    preview_every = param("preview_every", 1, int)
    inputs = {
        "model": ref(param("model_ref")),
        "pos": ref(param("positive")),
        "neg": ref(param("negative")),
        "latent": ref(param("latent")),
        "seed": param("seed", -1, int),
        "steps": param("steps", 28, int),
        "cfg": param("cfg", 4.5, float),
        "denoise": param("denoise", 1.0, float),
        "preview_every": preview_every,
    }
    print(f"[sd3-sampler] steps={inputs['steps']} cfg={inputs['cfg']} "
          f"denoise={inputs['denoise']} seed={inputs['seed']}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "sd3.sample", "inputs": inputs}, tok)
    print(f"[sd3-sampler] job={jid} submitted "
          f"(preview {'every ' + str(preview_every) + ' step' if preview_every > 0 else 'off'})",
          flush=True)

    def on_poll(view):
        if preview_every <= 0:
            return
        p = view.get("progress") or {}
        cur, tot = p.get("current", 0), p.get("total", 0)
        if cur > 0 and tot > 0:
            emit_preview(f"{url}/preview/{jid}", cur / tot, tok)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 7200, int),
                      poll=param("poll_interval", 5, int),
                      on_poll=on_poll)
    outs = result["outputs"]
    latent_id = outs["latent"].get("id")
    seed = outs["seed"].get("value")
    print(f"[sd3-sampler] done in {time.time()-t0:.1f}s -> latent={latent_id} "
          f"seed={seed}", flush=True)
    emit(latent=latent_id, seed=seed)


if __name__ == "__main__":
    main()
