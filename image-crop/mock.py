"""image-crop mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("image", "mock-image")
x = param("x", 0, cast=int)
y = param("y", 0, cast=int)
w = param("width", 0, cast=int)
h = param("height", 0, cast=int)
print(f"[crop][mock] image={iid} ({x},{y},{w}x{h})")
oid = f"mock-image-crop-out-{iid}"
emit(image=oid, width=64, height=64)
