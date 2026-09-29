"""flux-vae-load mock：生成伪 VAE id。"""
from flowx_client import emit, param

n = param("name", "vae")
oid = f"mock-flux-vae-{n}"
print(f"[flux-vae-load][mock] name={n} -> {oid}")
emit(vae=oid)
