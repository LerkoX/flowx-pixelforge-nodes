"""h3-text-encode mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

print("[h3-text-encode][mock] ok")
emit(cond="mock-h3-cond", image=param("image", ""),
     last_image=param("last_image", ""))
