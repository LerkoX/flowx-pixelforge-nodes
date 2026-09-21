"""upscale-model-load mock：不调服务，透传伪 upscale_model id。"""
from flowx_client import emit, param

uid = f"mock-upsm-{param('name', 'upscaler')}"
print(f"[upscale-model-load][mock] {param('name', '?')} -> {uid}")
emit(upscale_model=uid)
