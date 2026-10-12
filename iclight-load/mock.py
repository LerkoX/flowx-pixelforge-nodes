"""iclight-load mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_v = param("variant", "fc")
print("[iclight-load][mock] variant=%s dtype=%s weights=%s reload=%s"
      % (_v, param("dtype", "auto"), param("weights", ""), param("reload", "false")))
emit(model="iclight:" + str(_v), params_mb=4360, vram_mb=4360,
     info='{"variant":"%s","dtype":"fp32","conv_in":%d,"keys":686,"cache_hit":false}'
          % (_v, 8 if str(_v) == "fc" else 12))
