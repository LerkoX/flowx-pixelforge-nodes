"""video-gen：文/图生视频（Wan TI2V 系），走推理服务异步任务体系。

分钟级任务：POST /jobs 提交 video.sample 算子 → 轮询状态/进度 → 完成取 VIDEO 对象 ID。
预览：preview_every>0 且 Studio 注入 FLOWX_CALLBACK_URL 时，服务端采样循环
经回调通道向画布推进度卡片帧（3D latent 无法廉价投影，进度卡片零模型开销）。
可选接 load-image 上传后的 IMAGE 对象作为首帧（图生视频）；留空为纯文生视频。
节点被杀/超时不会自动取消服务端任务——超时会主动 /interrupt，强杀请手动调
POST /interrupt。
"""
import os
import time

from flowx_client import (emit, param, ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
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
    }
    image_id = param("image", "")  # 可选：首帧 IMAGE 对象 ID（图生视频）
    if image_id:
        inputs["image"] = ref(image_id)

    # 预览进度推送：把 Studio 注入的回调地址透传给推理服务（用 PUBLIC 地址，
    # 预览帧由远程推理服务推送，回环地址不可达）
    callback_url = (os.environ.get("FLOWX_CALLBACK_URL_PUBLIC")
                    or os.environ.get("FLOWX_CALLBACK_URL", ""))
    if callback_url:
        inputs["preview_callback_url"] = callback_url
        inputs["preview_token"] = os.environ.get("FLOWX_AUTH_TOKEN", "")
        inputs["preview_every"] = param("preview_every", 5, int)
        print(f"[video-gen] progress push enabled (every {inputs['preview_every']} steps)",
              flush=True)

    print(f"[video-gen] {inputs['width']}x{inputs['height']} "
          f"frames={inputs['num_frames']} fps={inputs['fps']} "
          f"steps={inputs['steps']} cfg={inputs['cfg']} seed={inputs['seed']} "
          f"i2v={'yes' if image_id else 'no'}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "video.sample", "inputs": inputs}, tok)
    print(f"[video-gen] job={jid} submitted", flush=True)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 7200, int),
                      poll=param("poll_interval", 5, int))
    outs = result["outputs"]
    video_id = outs["video"].get("id")
    seed = outs["seed"].get("value")
    print(f"[video-gen] done in {time.time()-t0:.1f}s -> video={video_id} seed={seed}",
          flush=True)
    emit(video=video_id, seed=seed)


if __name__ == "__main__":
    main()
