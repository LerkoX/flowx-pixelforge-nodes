"""video-enhance mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

_f = param("face_model", "gfpgan")
_u = param("upscale_model", "")
print("[video-enhance][mock] face_model=%s upscale_model=%s tile=%s"
      % (_f, _u, param("tile", 0, int)))
emit(video="mock-enhanced-video", faces=61, scale=2 if _u else 1)
