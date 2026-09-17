"""motion-loader mock：透传模型引用并标注运动模块。"""
from flowx_client import emit, param

model = param("model_ref", "mock-model")
motion = param("motion_name", "animatediff-motion-adapter-v1-5-2")
composed = f"{model}+motion:{motion}"
print(f"[motion][mock] {composed}")
emit(model_ref=composed, clip_ref=model, vae_ref=model)
