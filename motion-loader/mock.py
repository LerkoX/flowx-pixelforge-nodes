"""motion-loader mock：透传底模名并标注运动模块。"""
from flowx_client import emit, param

ckpt = param("ckpt_name", "mock-ckpt")
motion = param("motion_name", "animatediff-motion-adapter-v1-5-2")
composed = f"{ckpt}+motion:{motion}"
print(f"[motion][mock] {composed}")
emit(model_ref=composed, clip_ref=composed, vae_ref=composed)
