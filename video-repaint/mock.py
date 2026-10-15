"""video-repaint mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_video = param("video", "mock-video-obj")
print("[video-repaint][mock] video=%s model=%s pos=%s neg=%s strength=%s steps=%s "
      "cfg=%s seed=%s cn=%s preview_every=%s"
      % (_video, param("model", "mock-model"), param("pos", "mock-cond"),
         param("neg", "mock-uncond"), param("strength", "0.6"),
         param("steps", "20"), param("cfg", "7.0"), param("seed", "-1"),
         param("control_net", "") or "-", param("preview_every", "1")))
emit(video="mock-video-repaint-" + str(_video), seed=12345, num_frames=8)
