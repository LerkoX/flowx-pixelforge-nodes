"""ksampler：KSampler 采样，消费正/反 conditioning 与 latent，输出采样后 latent。

经推理服务异步任务通道执行（POST /jobs + 轮询）：分钟级采样不被 HTTP 空闲超时
掐断，且采样中服务端逐步把 latent 预览帧（JPEG）留在 GET /preview/{job_id}
（只留最新一帧）。本节点轮询进度时把帧地址经 stdout 标记（FLOWX_PREVIEW）上报
Studio，Studio 中转拉帧给画布实时渲染——等效 ComfyUI 采样器实时预览。
标记只是 URL，媒体本体全程 HTTP，不走 base64。
preview_every=0 关闭预览（进度仍经 job 轮询上报）。
"""
import time

from flowx_client import (emit, emit_preview, param, ref, submit_job, token,
                          wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    preview_every = param("preview_every", 1, int)
    inputs = {
        "model": ref(param("model_ref")),
        "pos": ref(param("positive")),
        "neg": ref(param("negative")),
        "latent": ref(param("latent")),
        "seed": param("seed", -1, int),
        "steps": param("steps", 20, int),
        "cfg": param("cfg", 7.0, float),
        "sampler_name": param("sampler_name", "euler"),
        "scheduler": param("scheduler", "normal"),
        "denoise": param("denoise", 1.0, float),
        "preview_every": preview_every,
    }
    print(f"[ksampler] steps={inputs['steps']} cfg={inputs['cfg']} "
          f"sampler={inputs['sampler_name']}+{inputs['scheduler']} "
          f"seed={inputs['seed']} denoise={inputs['denoise']}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "sample", "inputs": inputs}, tok)
    print(f"[ksampler] job={jid} submitted "
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
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 2, int),
                      on_poll=on_poll)
    outs = result["outputs"]
    latent_id = outs["latent"].get("id")
    seed = outs["seed"].get("value")
    print(f"[ksampler] done in {time.time()-t0:.1f}s -> latent={latent_id} "
          f"seed={seed}", flush=True)
    emit(latent=latent_id, seed=seed)


if __name__ == "__main__":
    main()
