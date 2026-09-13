"""lora-loader mock：透传模型引用并标注 LoRA。"""
from flowx_client import emit, param

model = param("model_ref", "mock-model")
lora = param("lora_name", "mock-lora")
strength = param("strength", 1.0, float)
patched = f"{model}+lora:{lora}@{strength}"
print(f"[lora][mock] {patched}")
emit(model_ref=patched, clip_ref=model)
