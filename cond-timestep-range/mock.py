"""mock：不真跑 GPU。"""
from flowx_client import emit, param

print("[cond-timestep-range][mock] skipped", flush=True)
emit(cond="mock-cond-with-window")
