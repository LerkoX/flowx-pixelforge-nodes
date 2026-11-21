"""qwen-image-vae-load mock：输出伪 vae id。"""
from flowx_client import emit, param
print(f"[qwen-image-vae-load][mock] model={param('model','Qwen-Image-2512')}", flush=True)
emit(vae="mock-qwen-vae")
