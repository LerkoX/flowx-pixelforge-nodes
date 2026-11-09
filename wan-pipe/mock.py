"""mock：不真跑 GPU，输出伪 model id。"""
from flowx_client import emit

print("[wan-pipe][mock] assembled (skipped)", flush=True)
emit(model="mock-wan-pipe-model")
