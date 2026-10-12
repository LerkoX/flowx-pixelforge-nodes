"""colorize-load mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_v = param("variant", "modelscope")
print("[colorize-load][mock] variant=%s dtype=%s reload=%s"
      % (_v, param("dtype", "fp32"), param("reload", "false")))
emit(model="ddcolor:" + str(_v), params_mb=310, vram_mb=1240,
     info='{"variant":"%s","dtype":"fp32","load":"strict","cache_hit":false}' % _v)
