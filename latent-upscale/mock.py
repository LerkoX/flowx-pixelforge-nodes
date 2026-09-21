"""latent-upscale mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

latent = f"mock-latent-{param('latent', 'x')}"
print(f"[latent-upscale][mock] ok")
emit(latent=latent)
