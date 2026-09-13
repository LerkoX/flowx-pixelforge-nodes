"""inference-ensure mock：不访问网络，直接回显 service_url。"""
from flowx_client import emit, param

url = param("service_url", "http://127.0.0.1:8100").rstrip("/")
print(f"[ensure][mock] service_url={url} (health check skipped)")
emit(service_url=url, status="ok", gpu_name="mock-gpu")
