"""sd3-vae-decode mock：产出假 IMAGE 引用。"""
from flowx_client import emit, param

print(f"[sd3-decode][mock] latent={param('latent', 'mock-latent')}")
emit(image="mock-sd3-image")
