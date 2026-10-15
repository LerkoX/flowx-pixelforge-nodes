"""frames-to-video mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_frames = param("frames", "mock-frames-obj")
_fps = param("fps", "24")
print("[frames-to-video][mock] frames=%s fps=%s" % (_frames, _fps))
emit(video="mock-video-obj-" + str(_frames), num_frames=8)
