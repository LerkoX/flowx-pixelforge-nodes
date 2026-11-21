"""qwen-image-te-load mock：输出伪 text_encoder id。"""
from flowx_client import emit, param
print(f"[qwen-image-te-load][mock] model={param('model','Qwen-Image-2512')}", flush=True)
emit(text_encoder="mock-qwen-text-encoder")
