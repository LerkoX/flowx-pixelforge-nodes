"""ipadapter-load mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

ipa = f"mock-ipadapter-{param('name', 'x')}"
print("[ipadapter-load][mock] ok")
emit(ipadapter=ipa)
