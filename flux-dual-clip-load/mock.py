"""flux-dual-clip-load mock：生成伪 CLIP id。"""
from flowx_client import emit, param

t = param("t5", "mock.gguf")
oid = f"mock-flux-clip-from-{t}"
print(f"[flux-dual-clip-load][mock] t5={t} -> {oid}")
emit(clip=oid)
