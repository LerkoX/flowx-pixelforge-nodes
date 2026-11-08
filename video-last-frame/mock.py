"""video-last-frame mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

_video = param("video", "mock-video-obj")
_off = param("offset", 0.2, float)
print("[video-last-frame][mock] video=%s offset=%s" % (_video, _off))
emit(image="mock-last-frame-" + str(_video), width=1344, height=768)
