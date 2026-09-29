"""flux-empty-latent mock：生成伪 LATENT id。"""
from flowx_client import emit, param

w = param("width", 768)
h = param("height", 768)
oid = f"mock-flux-latent-{w}x{h}"
print(f"[flux-empty-latent][mock] {w}x{h} -> {oid}")
emit(latent=oid)
