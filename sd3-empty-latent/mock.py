"""sd3-empty-latent mock：产出假 LATENT 引用。"""
from flowx_client import emit, param

w = param("width", 512, int)
h = param("height", 512, int)
b = param("batch_size", 1, int)
print(f"[sd3-latent][mock] {w}x{h} batch={b}")
emit(latent="mock-sd3-latent", info=f"latent=({b},16,{h//8},{w//8}) dtype=fp16")
