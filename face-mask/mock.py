"""face-mask mock：不调服务，透传伪蒙版 id 与伪坐标。"""
from flowx_client import emit, param

mid = f"mock-facemask-{param('image', 'img')}"
print(f"[face-mask][mock] expand={param('expand', '0.6')} feather={param('feather', '16')} -> {mid}")
emit(mask=mid, x=64, y=64, width=256, height=256)
