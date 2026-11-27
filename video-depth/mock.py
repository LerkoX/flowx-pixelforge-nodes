"""video-depth mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

_v = param("video", "mock-video-obj")
_inv = str(param("invert", False)).strip().lower() in ("1", "true", "yes", "on")
print("[video-depth][mock] video=%s invert=%s" % (_v, _inv))
emit(video="mock-depth-video-obj", depth="mock-depth-image-obj", fps=24)
