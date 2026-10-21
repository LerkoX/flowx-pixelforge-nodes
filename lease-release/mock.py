"""lease-release mock：不调服务，直接返回伪释放结果。"""
from flowx_client import emit, param

url = param("service_url", "http://mock-inference:8100")
print(f"[lease-release][mock] release lease on {url}")
emit(released="true", lease_id="mock-execution")
