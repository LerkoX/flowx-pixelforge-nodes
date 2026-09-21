"""clip-vision-load mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

cv = f"mock-clipvision-{param('name', 'x')}"
print("[clip-vision-load][mock] ok")
emit(clip_vision=cv)
