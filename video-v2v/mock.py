"""video-v2v mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_video = param("video", "mock-video-obj")
print("[video-v2v][mock] video=%s model=%s prompt=%r strength=%s steps=%s cfg=%s "
      "seed=%s frames=%s"
      % (_video, param("model", "mock-model"), param("prompt", ""),
         param("strength", "0.6"), param("steps", "25"), param("cfg", "7.5"),
         param("seed", "-1"), param("num_frames", "0")))
emit(video="mock-video-v2v-" + str(_video), seed=12345)
