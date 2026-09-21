"""cond-average mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

cond = f"mock-cond-{param('cond_a', 'x')}"
print(f"[cond-average][mock] ok")
emit(cond=cond)
