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
    # **model 可省略（2026-11-16 简化流水线）**：独立装配模式下算子内部
    # 自己 from_pretrained 全套组件，不再需要上游 wan-pipe。留空即走独立装配，
    # 让整条链缩成 LoadVideo→Pose→Mask→Animate→Save 五个节点。
    # 仍兼容旧图：给了 model 就透传（服务端 WAN_STANDALONE=0 时才会真正用到）。
    if not image_id:
        raise RuntimeError("wan-animate2: image（角色参考图）必填")
    if not pose_id:
        raise RuntimeError(
            "wan-animate2: pose_video 必填（建议 video-control-preprocess mode=pose）")

    # replace 模式（人物替换）：给 background_video + mask_video 即启用，
    # 保留原视频背景只换人（bg×(1-mask) 抹除在服务端算子里做）
    if model_id:
        inputs["model"] = ref(model_id)
    bg_id = param("background_video", "")
    mask_id = param("mask_video", "")
    inputs = {
        "image": ref(image_id),
        "pose_video": ref(pose_id),
        "mode": param("mode", "auto"),
        "inpaint_fill": param("inpaint_fill", "black"),
        "inpaint_erode": param("inpaint_erode", 0, int),
        "inpaint_expand": param("inpaint_expand", 0, int),
        "inpaint_telea_radius": param("inpaint_telea_radius", 5, int),
        "fps": param("fps", 0, int),
        "composite": param("composite", "0") in ("1", "true", "True"),
        "composite_expand": param("composite_expand", 6, int),
        "composite_feather": param("composite_feather", 20, int),
        "composite_gain": param("composite_gain", 1.25, float),
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
    _face = param("face_video", "")
    if _face:
        inputs["face_video"] = ref(_face)
    # **LoRA / 打光 / 预计算文本向量（2026-11-16）**
    for _k in ("lora", "relight_lora", "prompt_embeds_path",
               "negative_embeds_path"):
        _v = param(_k, "")
        if _v:
            inputs[_k] = _v
    inputs["relight_scale"] = param("relight_scale", 1.0, float)
    if bg_id:
        inputs["background_video"] = ref(bg_id)
    if mask_id:
        inputs["mask_video"] = ref(mask_id)

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
