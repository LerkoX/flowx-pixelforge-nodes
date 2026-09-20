"""sd3-vae-encode mock：产出假 LATENT 引用。"""
from flowx_client import emit, param

print(f"[sd3-encode-img][mock] image={param('image', 'mock-image')}")
emit(latent="mock-sd3-latent-encoded")
