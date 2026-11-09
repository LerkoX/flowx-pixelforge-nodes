"""mock：不真跑 GPU，输出伪组件 id。"""
from flowx_client import emit

print("[wan-image-encoder-load][mock] skipped", flush=True)
emit(image_encoder="mock-wan-image-encoder-load")
