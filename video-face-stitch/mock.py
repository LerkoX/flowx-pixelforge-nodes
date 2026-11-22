"""video-face-stitch mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

_v = param("video", "mock-video-obj")
_c = param("crop_video", "mock-crop-obj")
print("[video-face-stitch][mock] video=%s crop=%s" % (_v, _c))
emit(video="mock-video-out", fps=24, num_frames=141)
