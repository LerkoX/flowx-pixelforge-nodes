"""wan-animate：Wan-Animate 角色动画 / 角色替换（Wan2.2-Animate-14B）。

自包含节点：服务端算子 wan.animate 由 server_op.py 自注册（ensure_plugin）。

两种模式（mode）：
- animate  角色动画：给一张角色图 + 参考视频，让角色复现参考视频的动作与表情
- replace  角色替换：把背景视频里的人换成新角色（额外需 background_video +
           mask_video，mask 白=生成新内容 / 黑=保留原画面）

⚠ pose_video / face_video 必须是**预处理后**的骨骼与人脸表示，不能直接喂
原始视频——预处理脚本见官方 Wan2.2 仓库（humanaigc/Wan-Animate 的
preprocess）。diffusers 尚未内置该预处理。

与 MiniMax-H3 + SAM3 手工 inpaint 方案相比，Wan-Animate 是**为此任务专训**
的端到端模型（含 Relighting LoRA 做环境光照/色调融合），预期效果更好。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "wan.animate", tok=tok)

    model_id = param("model", "")
    if not model_id:
        raise RuntimeError("wan-animate: model 必填（接 wan-load task=animate 的 model）")

    image_id = param("image", "")
    pose_id = param("pose_video", "")
    face_id = param("face_video", "")
    bg_id = param("background_video", "")
    mask_id = param("mask_video", "")
    mode = param("mode", "animate")

    if not image_id:
        raise RuntimeError("wan-animate: image（角色参考图）必填")
    if not pose_id or not face_id:
        raise RuntimeError(
            "wan-animate: pose_video 与 face_video 必填"
            "（须为预处理后的骨骼/人脸视频）")
    if mode == "replace" and (not bg_id or not mask_id):
        raise RuntimeError(
            "wan-animate: replace 模式需 background_video 与 mask_video"
            "（白=生成区 / 黑=保留区）")

    inputs = {
        "model": ref(model_id),
        "image": ref(image_id),
        "pose_video": ref(pose_id),
        "face_video": ref(face_id),
        "prompt": param("prompt", ""),
        "negative_prompt": param("negative_prompt", ""),
        "mode": mode,
        "segment_frame_length": param("segment_frame_length", 77, int),
        "height": param("height", 0, int),
        "width": param("width", 0, int),
        "steps": param("steps", 20, int),
        "guidance_scale": param("guidance_scale", 1.0, float),
        "prev_segment_conditioning_frames":
            param("prev_segment_conditioning_frames", 1, int),
        "motion_encode_batch_size":
            param("motion_encode_batch_size", 0, int),
        "seed": param("seed", -1, int),
    }
    if bg_id:
        inputs["background_video"] = ref(bg_id)
    if mask_id:
        inputs["mask_video"] = ref(mask_id)

    t0 = time.time()
    jid = submit_job(url, {"name": "wan.animate", "inputs": inputs}, tok)

    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[wan-animate] job={jid} mode={mode} "
          f"seg={inputs['segment_frame_length']} steps={inputs['steps']} "
          f"seed={inputs['seed']}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    fps = out.get("fps") or 30
    print(f"[wan-animate] -> video={out['video']} seed={out['seed']} "
          f"fps={fps} ({time.time()-t0:.1f}s)", flush=True)
    emit(video=out["video"], seed=out["seed"], fps=fps)


if __name__ == "__main__":
    main()
