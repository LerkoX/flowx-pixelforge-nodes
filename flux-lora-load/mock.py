"""flux-lora-load mock：透传伪 MODEL id。"""
from flowx_client import emit, param

m = param("model", "mock-flux-unet")
lora = param("lora_name", "mock-lora")
oid = f"{m}+lora({lora})"
print(f"[flux-lora-load][mock] model={m} lora={lora} -> {oid}")
emit(model=oid, info=f"mock adapter={lora}")
