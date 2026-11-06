"""beautify-batch mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[beautify-batch][mock] dir=%s crop=%s whiten=%s eye=%s slim=%s"
      % (param("dir", "."), param("crop_size", "800"), param("whiten", "0.5"),
         param("eye", "0.12"), param("slim", "0.10")))
emit(images='@["mock-b1","mock-b2","mock-b3"]', count=3,
     names='@["a.jpg","b.jpg","c.jpg"]', dir="/mock/dir")
