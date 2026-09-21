"""controlnet-load mock：不调服务，透传伪 control_net id。"""
from flowx_client import emit, param

cid = f"mock-cn-{param('name', 'controlnet')}"
print(f"[controlnet-load][mock] {param('name', '?')} -> {cid}")
emit(control_net=cid)
