"""catvton-tryon mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_p = param("person", "mock-person")
print("[catvton-tryon][mock] person=%s cloth=%s mask=%s steps=%s guidance=%s "
      "repaint=%s mask_blur=%s"
      % (_p, param("cloth", "mock-cloth"), param("mask", "mock-mask"),
         param("steps", "50"), param("guidance", "2.5"),
         param("repaint", "true"), param("mask_blur", "9")))
emit(image="mock-tryon-" + str(_p), width=768, height=1280,
     info='{"variant":"mix","dtype":"fp32","chunk":1024,"steps":50,"guidance":2.5,'
          '"mask_blur":9,"repaint":true,"mask_coverage":0.52,"seconds":268.3,'
          '"peak_vram_gb":6.24}')
