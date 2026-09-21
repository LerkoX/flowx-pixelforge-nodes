"""vae-load mock：不调服务，透传伪 vae id。"""
from flowx_client import emit, param

vid = f"mock-vae-{param('name', 'vae')}"
print(f"[vae-load][mock] {param('name', '?')} -> {vid}")
emit(vae=vid)
