"""sam-load mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[sam-load][mock] variant=%s dtype=%s" % (param("variant", "vit_l"),
                                                param("dtype", "fp32")))
emit(model="sam:mock", params_mb=1191, vram_mb=1191)
