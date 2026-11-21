"""qwen-image-transformer-load mock：输出伪 transformer id。"""
from flowx_client import emit, param
print(f"[qwen-image-transformer-load][mock] model={param('model','Qwen-Image-2512')}", flush=True)
emit(transformer="mock-qwen-transformer")
