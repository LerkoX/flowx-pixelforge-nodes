"""video-gen：文/图生视频（Wan TI2V 系），走推理服务异步任务体系。

分钟级任务：POST /jobs 提交 video.sample 算子 → 轮询状态/进度 → 完成取 VIDEO 对象 ID。
预览：preview_every>0 时服务端采样循环把进度卡片帧（3D latent 无法廉价投影，
进度卡片零模型开销）留在 GET /preview/{job_id}；本节点轮询时把帧地址经 stdout
标记（FLOWX_PREVIEW）上报 Studio，Studio 中转拉帧给画布——媒体全程 HTTP，
不走 base64。
可选接 load-image 上传后的 IMAGE 对象作为首帧（图生视频）；留空为纯文生视频。
节点被杀/超时不会自动取消服务端任务——超时会主动 /interrupt，强杀请手动调
POST /interrupt。
"""
import time

from flowx_client import (emit, emit_preview, param, ref, submit_job, token,
                          wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    preview_every = param("preview_every", 5, int)
    output_mode = param("output_mode", "pil")  # pil=采样+解码一体; latent=只出 latent 交下游 vae.decode_video
    inputs = {
        "model": ref(param("model_ref")),
        "prompt": param("prompt"),
        "neg_prompt": param("negative_prompt", ""),
        "width": param("width", 832, int),
        "height": param("height", 480, int),
        "num_frames": param("num_frames", 121, int),
        "fps": param("fps", 24, int),
        "steps": param("steps", 50, int),
        "cfg": param("cfg", 5.0, float),
        "seed": param("seed", -1, int),
        "decode_chunk_size": param("decode_chunk_size", 0, int),
        "preview_every": preview_every,
    }
    image_id = param("image", "")  # 可选：首帧 IMAGE 对象 ID（图生视频）
    if image_id:
        inputs["image"] = ref(image_id)
    # latent 模式不解码，分块参数无意义（算子未声明的输入会被静默忽略，仅日志不展示）
    op_name = "video.sample_latent" if output_mode == "latent" else "video.sample"

    print(f"[video-gen] {inputs['width']}x{inputs['height']} "
          f"frames={inputs['num_frames']} fps={inputs['fps']} "
          f"steps={inputs['steps']} cfg={inputs['cfg']} seed={inputs['seed']} "
          f"i2v={'yes' if image_id else 'no'} mode={output_mode} "
          f"preview={'every ' + str(preview_every) + ' steps' if preview_every > 0 else 'off'}",
          flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": op_name, "inputs": inputs}, tok)
    print(f"[video-gen] job={jid} submitted ({op_name})", flush=True)

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
    seed = outs["seed"].get("value")
    if output_mode == "latent":
        latent_id = outs["latent"].get("id")
        print(f"[video-gen] done in {time.time()-t0:.1f}s -> latent={latent_id} seed={seed}",
              flush=True)
        emit(latent=latent_id, seed=seed)
        return
    video_id = outs["video"].get("id")
    print(f"[video-gen] done in {time.time()-t0:.1f}s -> video={video_id} seed={seed}",
          flush=True)
    emit(video=video_id, seed=seed)


if __name__ == "__main__":
    main()
