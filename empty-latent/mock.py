"""empty-latent mock：生成伪 latent_id。"""
from flowx_client import emit, param

w = param("width", 512, int)
h = param("height", 512, int)
b = param("batch_size", 1, int)
lid = f"mock-latent-{w}x{h}x{b}"
print(f"[latent][mock] {w}x{h} batch={b} -> {lid}")
emit(latent=lid)
