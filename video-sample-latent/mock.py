"""video-sample-latent mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

latent = f"mock-latent-{param('model', 'x')}"
seed = f"mock-seed-{param('model', 'x')}"
print(f"[video-sample-latent][mock] ok")
emit(latent=latent, seed=seed)
