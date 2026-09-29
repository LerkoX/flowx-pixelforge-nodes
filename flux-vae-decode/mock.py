"""flux-vae-decode mock：生成伪 IMAGE id。"""
from flowx_client import emit, param

l = param("latent", "mock")
oid = f"mock-flux-image-from-{l}"
print(f"[flux-vae-decode][mock] latent={l} -> {oid}")
emit(image=oid)
