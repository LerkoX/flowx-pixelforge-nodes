"""mock：不真跑 GPU，输出伪组件 id。"""
from flowx_client import emit

print("[wan-transformer-load][mock] skipped", flush=True)
emit(transformer="mock-wan-transformer-load")
