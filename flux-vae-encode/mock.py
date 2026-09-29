"""flux-vae-encode mock：生成伪 LATENT id。"""
from flowx_client import emit, param

i = param("image", "mock")
oid = f"mock-flux-latent-from-{i}"
print(f"[flux-vae-encode][mock] image={i} -> {oid}")
emit(latent=oid)
