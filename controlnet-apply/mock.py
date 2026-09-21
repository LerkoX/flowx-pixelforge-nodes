"""controlnet-apply mock：不调服务，透传伪 control id。"""
from flowx_client import emit, param

cid = f"mock-control-{param('control_net', 'cn')}"
print(f"[controlnet-apply][mock] strength={param('strength', '1.0')} -> {cid}")
emit(control=cid)
