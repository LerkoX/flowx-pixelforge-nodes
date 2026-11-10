"""mock。"""
from flowx_client import emit
print("[cond-pack][mock] skipped", flush=True)
emit(cond="mock-cond-packed")
