"""image-rotate mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("image", "mock-image")
angle = param("angle", 0.0, cast=float)
expand = param("expand", "true")
resample = param("resample", "bicubic")
print(f"[rotate][mock] image={iid} angle={angle} expand={expand} resample={resample}")
oid = f"mock-image-rotate-out-{iid}"
emit(image=oid, width=64, height=64)
