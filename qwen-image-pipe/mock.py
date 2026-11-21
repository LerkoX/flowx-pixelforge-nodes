"""qwen-image-pipe mock：输出伪 model id。"""
from flowx_client import emit, param
print(f"[qwen-image-pipe][mock] lora={param('lora','')}", flush=True)
emit(model="mock-qwen-pipe")
