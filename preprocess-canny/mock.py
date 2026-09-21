"""preprocess-canny mock：不调服务，直接透传伪 edges id。"""
from flowx_client import emit, param

iid = f"mock-edges-{param('image', 'mock-image')}"
print(f"[canny][mock] {param('image', 'mock-image')} -> {iid}")
emit(image=iid)
