"""video-concat mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

_videos = param("videos", "mock-a,mock-b")
_n = len([s for s in _videos.split(",") if s.strip()])
print("[video-concat][mock] videos=%s crossfade=%s segments=%d"
      % (_videos, param("crossfade", 0.5, float), _n))
emit(file_path="/tmp/mock-long-video.mp4", duration=15.6, segments=_n)
