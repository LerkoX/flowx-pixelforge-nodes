"""preprocess-depth mock：不调服务，透传伪深度图 id。"""
from flowx_client import emit, param

iid = f"mock-depth-{param('image', 'img')}"
print(f"[depth][mock] detect_resolution={param('detect_resolution', '512')} -> {iid}")
emit(image=iid)
