"""embedding-load mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

clip = f"mock-clip-{param('clip', 'x')}"
print(f"[embedding-load][mock] ok")
emit(clip=clip)
