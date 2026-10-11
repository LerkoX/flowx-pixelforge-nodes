"""grounding-dino-load mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[grounding-dino-load][mock] variant=%s dtype=%s"
      % (param("variant", "tiny"), param("dtype", "fp32")))
emit(model="gd:mock", params_mb=657, vram_mb=657)
