"""checkpoint-loader mock：不加载模型，回显 ckpt 名作为引用。"""
from flowx_client import emit, param

ckpt = param("ckpt_name", "mock-model")
print(f"[loader][mock] ckpt={ckpt} (load skipped)")
emit(model_ref=ckpt, clip_ref=ckpt, vae_ref=ckpt)
