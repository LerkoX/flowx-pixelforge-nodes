"""ksampler-brushnet mock：模拟采样耗时，输出伪 latent_id。"""
import random

from flowx_client import emit, param

seed = param("seed", -1, int)
steps = param("steps", 20, int)
if seed < 0:
    seed = random.randint(0, 2**32 - 1)
print(f"[ksampler-brushnet][mock] steps={steps} seed={seed} (sampling skipped)")
emit(latent=f"mock-brushnet-sampled-{seed}", seed=seed)
