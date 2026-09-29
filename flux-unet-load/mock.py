"""flux-unet-load mock：生成伪 MODEL id。"""
from flowx_client import emit, param

t = param("transformer", "mock.gguf")
oid = f"mock-flux-unet-from-{t}"
print(f"[flux-unet-load][mock] transformer={t} -> {oid}")
emit(model=oid)
