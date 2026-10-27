"""vae-encode mock：生成伪 latent_id。"""
from flowx_client import emit, param

image = param("image", "mock")
lid = f"mock-latent-from-{image}"
print(f"[encode][mock] image={image} -> {lid}")
emit(latent=lid)
