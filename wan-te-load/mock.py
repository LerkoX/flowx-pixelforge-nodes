"""mock：不真跑 GPU，输出伪组件 id。"""
from flowx_client import emit

print("[wan-te-load][mock] skipped", flush=True)
emit(text_encoder="mock-wan-te-load")
