"""minimax-h3-load mock：模拟装配耗时，输出伪 model id。"""
import time

from flowx_client import emit, param

tr = param("transformer", "minimax_h3_fl2va_int8_convrot.safetensors")
lora = param("lora", "")
print(f"[minimax-h3-load][mock] loading transformer={tr} lora={lora or 'none'}",
      flush=True)
time.sleep(0.2)
print("[minimax-h3-load][mock] pipeline assembled (skipped)")
emit(model="mock-minimax-h3-model")
