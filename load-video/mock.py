"""load-video mock：不访问服务，验证参数注入与输出提取（链路打通用）。

真机路径是「上传原片 → video.load 算子解码」，mock 只回一个假的 VIDEO 对象 ID。
"""
from flowx_client import emit, param

_path = param("video_path", "~/flowx-input/mock.mp4")
print("[load-video][mock] path=%s start=%s end=%s stride=%s max_frames=%s "
      "fps=%s %sx%s"
      % (_path, param("start", "0"), param("end", "0"), param("stride", "1"),
         param("max_frames", "0"), param("fps", "0"),
         param("width", "0"), param("height", "0")))
emit(video="mock-video-obj", fps=24, num_frames=121, width=832, height=480)
