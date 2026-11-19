"""ksampler-brushnet：BrushNet 局部重绘采样（sd.sample_brushnet）。

经推理服务异步任务通道执行（POST /jobs + 轮询）：分钟级采样不被 HTTP 空闲
超时掐断；采样中服务端逐步把 latent 预览帧（JPEG）留在 GET /preview/{job_id}
（只留最新一帧），节点轮询进度时把帧地址经 stdout 标记（FLOWX_PREVIEW）上报
Studio，Studio 中转拉帧给画布实时渲染——等效 ComfyUI 采样器实时预览。
preview_every=0 关闭预览（进度仍经 job 轮询上报）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base,
                          param, ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)  # 画布预览帧由 Studio 取，必须宿主机可达
    tok = token()
    ensure_plugin(url, "sd.sample_brushnet", tok=tok)
    preview_every = param("preview_every", 1, int)
    inputs = {
        "model": ref(param("model_ref")),
        "pos": ref(param("positive")),
        "neg": ref(param("negative")),
        "latent": ref(param("latent")),
        "control": ref(param("control")),
        "seed": param("seed", -1, int),
        "steps": param("steps", 20, int),
        "cfg": param("cfg", 7.0, float),
        "sampler_name": param("sampler_name", "euler"),
        "scheduler": param("scheduler", "normal"),
        "denoise": param("denoise", 1.0, float),
        "preview_every": preview_every,
    }
    # 可选：brushnet 分支专用 cond（PowerPaint task prompt）
    bpos = param("brushnet_pos", "")
    if bpos:
        inputs["brushnet_pos"] = ref(bpos)
    bneg = param("brushnet_neg", "")
    if bneg:
        inputs["brushnet_neg"] = ref(bneg)
    # 可选：常规 ControlNet 链（与 BrushNet 正交叠加）
    cn = param("cn_control", "")
    if cn:
        inputs["cn_control"] = ref(cn)
    print(f"[ksampler-brushnet] steps={inputs['steps']} cfg={inputs['cfg']} "
          f"sampler={inputs['sampler_name']}+{inputs['scheduler']} "
          f"seed={inputs['seed']} denoise={inputs['denoise']} "
          f"task_cond={'yes' if bpos else 'no'} cn={'yes' if cn else 'no'}",
          flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "sd.sample_brushnet", "inputs": inputs}, tok)
    print(f"[ksampler-brushnet] job={jid} submitted "
          f"(preview {'every ' + str(preview_every) + ' step' if preview_every > 0 else 'off'})",
          flush=True)

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
                      poll=param("poll_interval", 2, int),
                      on_poll=on_poll)
    outs = result["outputs"]
    latent_id = outs["latent"].get("id")
    seed = outs["seed"].get("value")
    print(f"[ksampler-brushnet] done in {time.time()-t0:.1f}s -> "
          f"latent={latent_id} seed={seed}", flush=True)
    emit(latent=latent_id, seed=seed)


if __name__ == "__main__":
    main()
