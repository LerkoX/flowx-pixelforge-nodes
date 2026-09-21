"""upscale-model-apply mock：不调服务，透传伪 image id。"""
from flowx_client import emit, param

iid = f"mock-upscaled-{param('image', 'img')}"
print(f"[upscale-model-apply][mock] tile={param('tile', '0')} -> {iid}")
emit(image=iid)
