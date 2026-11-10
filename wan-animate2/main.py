"""wan-animate2：Wan-Animate-2 角色动画/替换（**不需要 face_video**）。

与 v1 的差别：v1 要 pose_video + face_video 两个预处理产物；Animate-2 只要
**角色图 + 动作视频（骨架）**，链路短一半。

推荐编排（真机教训：RGB 直接喂是「软条件」，画面会跳）：
  load-video ──→ video-control-preprocess(mode=pose) ──→ wan-animate2
  load-image(角色图) ────────────────────────────────┘
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()
    ensure_plugin(url, "wan.animate2", tok=tok)

    model_id = param("model", "")
    image_id = param("image", "")
    pose_id = param("pose_video", "")
    if not model_id:
        raise RuntimeError("wan-animate2: model 必填（接 wan-load task=animate2）")
    if not image_id:
        raise RuntimeError("wan-animate2: image（角色参考图）必填")
    if not pose_id:
        raise RuntimeError(
            "wan-animate2: pose_video 必填（建议 video-control-preprocess mode=pose）")

    inputs = {
        "model": ref(model_id), "image": ref(image_id),
        "pose_video": ref(pose_id),
        "prompt": param("prompt", ""),
        "negative_prompt": param("negative_prompt", ""),
        "pose_strength": param("pose_strength", 1.0, float),
        "pose_start_percent": param("pose_start_percent", 0.0, float),
        "pose_end_percent": param("pose_end_percent", 1.0, float),
        "reference_image_strength": param("reference_image_strength", 1.0, float),
        "segment_frame_length": param("segment_frame_length", 81, int),
        "height": param("height", 0, int),
        "width": param("width", 0, int),
        "steps": param("steps", 20, int),
        "guidance_scale": param("guidance_scale", 1.0, float),
        "seed": param("seed", -1, int),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "wan.animate2", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[wan-animate2] job={jid} steps={inputs['steps']} "
          f"pose_str={inputs['pose_strength']} ref_str="
          f"{inputs['reference_image_strength']}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    fps = out.get("fps") or 16
    print(f"[wan-animate2] -> video={out['video']} seed={out['seed']} fps={fps} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(video=out["video"], seed=out["seed"], fps=fps)


if __name__ == "__main__":
    main()
