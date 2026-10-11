"""image-composite mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("background", "mock-background")
fid = param("foreground", "mock-fg")
mid = param("mask", "")
x = param("x", 0, cast=int)
y = param("y", 0, cast=int)
print(f"[composite][mock] bg={iid} fg={fid} mask={mid or '-'} at ({x},{y})")
oid = f"mock-image-composite-out-{iid}"
emit(image=oid, width=64, height=64)
