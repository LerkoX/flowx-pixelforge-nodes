"""wan-vace：Wan-VACE 可控生成（**人物替换首选**）。

与 wan-animate 的区别：
- wan-animate 需要 pose_video + face_video（要预处理骨骼/人脸视频）
- **wan-vace 只要「原视频 + 遮罩 + 新角色图」**，不需要任何 pose 预处理
  → 可直接吃 video-person-mask（SAM3）的产物，链路更短、更不易失败

用法（人物替换）：
  load-video(原片) ─┐
  video-person-mask ─┼→ wan-vace → save-video
  load-image(角色图) ┘
遮罩语义：白=重绘区 / 黑=保留原画面（与 SAM3 输出一致）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "wan.vace", tok=tok)

    model_id = param("model", "")
    video_id = param("video", "")
    if not model_id:
        raise RuntimeError("wan-vace: model 必填（接 wan-load task=vace 的 model）")
    if not video_id:
        raise RuntimeError("wan-vace: video（原视频）必填")

    inputs = {
        "model": ref(model_id),
        "video": ref(video_id),
        "prompt": param("prompt", ""),
        "negative_prompt": param("negative_prompt", ""),
        "conditioning_scale": param("conditioning_scale", 1.0, float),
        "num_frames": param("num_frames", 0, int),
        "height": param("height", 0, int),
        "width": param("width", 0, int),
        "steps": param("steps", 30, int),
        "guidance_scale": param("guidance_scale", 5.0, float),
        "guidance_scale_2": param("guidance_scale_2", 0.0, float),
        "seed": param("seed", -1, int),
    }
    mask_id = param("mask", "")
    if mask_id:
        inputs["mask"] = ref(mask_id)
    ref_id = param("reference_image", "")
    if ref_id:
        inputs["reference_image"] = ref(ref_id)

    t0 = time.time()
    jid = submit_job(url, {"name": "wan.vace", "inputs": inputs}, tok)

    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[wan-vace] job={jid} frames={inputs['num_frames']} "
          f"steps={inputs['steps']} cond={inputs['conditioning_scale']} "
          f"ref={'yes' if ref_id else 'no'} mask={'yes' if mask_id else 'no'}",
          flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    fps = out.get("fps") or 16
    print(f"[wan-vace] -> video={out['video']} seed={out['seed']} fps={fps} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(video=out["video"], seed=out["seed"], fps=fps)


if __name__ == "__main__":
    main()
