"""vae-decode-video mock：不调服务，透传伪输出 id。"""
from flowx_client import emit, param

video = f"mock-video-{param('vae', 'x')}"
print(f"[vae-decode-video][mock] ok")
emit(video=video)
