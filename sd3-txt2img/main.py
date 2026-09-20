"""sd3-txt2img：SD3.5 文生图（MMDiT 新架构），消费模型引用与提示词，输出 IMAGE 对象 ID。

自包含节点：服务端算子 sd3.txt2img 由 server_op.py 随本节点包自注册
（ensure_plugin：/ops 比对 hash，缺失/不符自动上传热加载，服务端无需部署）。

经推理服务异步任务通道执行（POST /jobs + 轮询）：分钟~小时级采样不被 HTTP
空闲超时掐断。16ch latent 无法廉价投影成图，preview_every>0 时服务端留
进度卡片帧在 GET /preview/{job_id}，本节点轮询时把帧地址经 stdout 标记
（FLOWX_PREVIEW）上报 Studio 中转拉帧——进度可视化，与 ksampler 同模式。

模型经 checkpoint-loader 加载（默认弃 T5-XXL，CLIP-L/G 保底）；
本算子不与 SD1.x 的 clip/latent 节点混用（接线只认 model_ref）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, param, ref,
                          submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "sd3.txt2img", tok=tok)  # 服务端算子自注册（幂等）

    preview_every = param("preview_every", 1, int)
    inputs = {
        "model": ref(param("model_ref")),
        "prompt": param("prompt"),
        "negative_prompt": param("negative_prompt", ""),
        "width": param("width", 512, int),
        "height": param("height", 512, int),
        "steps": param("steps", 28, int),
        "cfg": param("cfg", 4.5, float),
        "seed": param("seed", -1, int),
        "preview_every": preview_every,
    }
    print(f"[sd3] {inputs['width']}x{inputs['height']} steps={inputs['steps']} "
          f"cfg={inputs['cfg']} seed={inputs['seed']}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "sd3.txt2img", "inputs": inputs}, tok)
    print(f"[sd3] job={jid} submitted "
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
    image_id = outs["image"].get("id")
    seed = outs["seed"].get("value")
    print(f"[sd3] done in {time.time()-t0:.1f}s -> image={image_id} "
          f"seed={seed}", flush=True)
    emit(image=image_id, seed=seed)


if __name__ == "__main__":
    main()
