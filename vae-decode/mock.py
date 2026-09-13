"""vae-decode mock：生成伪 image_id。"""
from flowx_client import emit, param

latent = param("latent", "mock")
iid = f"mock-image-from-{latent}"
print(f"[decode][mock] latent={latent} -> {iid}")
emit(image=iid)
