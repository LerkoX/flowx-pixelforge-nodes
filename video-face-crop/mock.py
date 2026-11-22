"""video-face-crop mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

_v = param("video", "mock-video-obj")
_m = param("mask", "mock-mask-obj")
print("[video-face-crop][mock] video=%s mask=%s" % (_v, _m))
emit(crop="mock-crop-obj", mask_white="mock-white-obj",
     boxes='[{"x":0,"y":0,"w":300,"h":300,"face_h":120}]',
     face_ref="mock-image-obj", num_frames=141, fps=24, face_h_max=120)
