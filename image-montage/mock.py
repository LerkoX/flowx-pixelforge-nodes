"""image-montage mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[image-montage][mock] images=%s cols=%s pad=%s label=%s"
      % (str(param("images"))[:40], param("cols", 0), param("pad", 10),
         param("label", "none")))
emit(image="mock-montage", width=780, height=520, grid='@[3, 2]', count=6)
