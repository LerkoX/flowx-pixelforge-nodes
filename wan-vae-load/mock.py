"""mock：不真跑 GPU，输出伪组件 id。"""
from flowx_client import emit

print("[wan-vae-load][mock] skipped", flush=True)
emit(vae="mock-wan-vae-load")
