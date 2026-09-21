"""latent-composite mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

latent = f"mock-latent-{param('dst', 'x')}"
print(f"[latent-composite][mock] ok")
emit(latent=latent)
