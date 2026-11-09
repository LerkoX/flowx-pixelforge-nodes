"""wan-load mock：模拟装配耗时，输出伪 model id。"""
import time

from flowx_client import emit, param

task = param("task", "t2v")
tr = param("transformer", "wan2.1_t2v_14B_bf16.safetensors")
tr2 = param("transformer2", "")
lora = param("lora", "")
print(f"[wan-load][mock] task={task} transformer={tr} "
      f"moe={bool(tr2)} lora={lora or 'none'}", flush=True)
time.sleep(0.2)
print("[wan-load][mock] pipeline assembled (skipped)", flush=True)
emit(model=f"mock-wan-{task}-model")
