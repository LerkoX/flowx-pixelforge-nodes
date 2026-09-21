"""cond-set-area mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

cond = f"mock-cond-{param('cond', 'x')}"
print(f"[cond-set-area][mock] ok")
emit(cond=cond)
