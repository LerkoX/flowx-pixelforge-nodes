"""video-to-frames mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_video = param("video", "mock-video-obj")
print("[video-to-frames][mock] video=%s start=%s end=%s stride=%s max_frames=%s"
      % (_video, param("start", "0"), param("end", "0"), param("stride", "1"),
         param("max_frames", "0")))
emit(frames="mock-frames-obj-" + str(_video), fps=24, count=8)
