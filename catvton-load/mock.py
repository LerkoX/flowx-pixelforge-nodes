"""catvton-load mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_v = param("variant", "mix")
print("[catvton-load][mock] variant=%s dtype=%s vae_dtype=%s chunk=%s"
      % (_v, param("dtype", "fp32"), param("vae_dtype", "fp32"),
         param("chunk", "1024")))
emit(vton="catvton:%s:fp32:fp32:1024" % _v,
     info='{"variant":"%s","unet":"fp32","vae":"fp32","chunk":1024,"load":161.0}'
          % _v)
