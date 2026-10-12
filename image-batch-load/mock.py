"""image-batch-load mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[image-batch-load][mock] dir=%s pattern=%s sort=%s limit=%s"
      % (param("dir", "~/flowx-input"), param("pattern", "*.png"),
         param("sort", "name"), param("limit", 0)))
emit(images='@["id-1", "id-2", "id-3"]', count=3,
     names='@["a.png", "b.png", "c.png"]', dir="/tmp")
