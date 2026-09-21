"""ipadapter-apply mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

m = f"mock-model-ipa-{param('model', 'x')}"
print("[ipadapter-apply][mock] ok")
emit(model=m)
