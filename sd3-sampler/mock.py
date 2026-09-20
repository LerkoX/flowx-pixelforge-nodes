"""sd3-sampler mock：不起采样，回显参数产出假 LATENT 引用。"""
from flowx_client import emit, param

steps = param("steps", 28, int)
cfg = param("cfg", 4.5, float)
denoise = param("denoise", 1.0, float)
print(f"[sd3-sampler][mock] steps={steps} cfg={cfg} denoise={denoise}")
emit(latent="mock-sd3-latent-sampled", seed="12345")
