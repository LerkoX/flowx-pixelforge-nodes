"""inference-ensure mock：不调服务，直接输出就绪态（含显存字段）。"""
from flowx_client import emit, param

url = param("service_url", "http://mock-inference:8100")
print(f"[ensure][mock] ready: {url}")
emit(service_url=url, status="ok", gpu_name="mock-gpu", offload_mode="none",
     vram_free_mb=7000, vram_total_mb=8192, resident_models="(empty)")
