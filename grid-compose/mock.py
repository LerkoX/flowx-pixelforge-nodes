"""grid-compose mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[grid-compose][mock] images=%s cell=%s gap=%s"
      % (param("images", "[]"), param("cell_size", "640"), param("gap", "10")))
emit(image="mock-grid-3x3", width=1940, height=1940, count=9)
