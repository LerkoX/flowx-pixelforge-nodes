"""video-person-mask mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

_p = param("prompt", "person")
_v = param("video", "mock-video-obj")
print("[video-person-mask][mock] prompt=%s video=%s" % (_p, _v))
emit(mask="mock-mask-obj", num_frames=117, width=1344, height=768, objects=1)
