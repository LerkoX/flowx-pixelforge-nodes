"""latent-set-noise-mask mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

latent = f"mock-latent-{param('latent', 'x')}"
print(f"[latent-set-noise-mask][mock] ok")
emit(latent=latent)
