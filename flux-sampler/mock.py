"""flux-sampler mock：生成伪采样 latent id。"""
from flowx_client import emit, param

latent = param("latent", "mock")
seed = param("seed", -1)
oid = f"mock-flux-sampled-from-{latent}"
print(f"[flux-sampler][mock] latent={latent} seed={seed} -> {oid}")
emit(latent=oid, seed=str(seed if int(seed) >= 0 else 42))
