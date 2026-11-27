"""preprocess-face mock：不调服务，直接透传伪 face id。"""
from flowx_client import emit, param

iid = f"mock-face-{param('image', 'mock-image')}"
print(f"[preprocess-face][mock] {param('image', 'mock-image')} -> {iid}")
emit(image=iid)
