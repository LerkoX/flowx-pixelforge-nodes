"""mask-feather mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("mask", "mock-mask")
r = param("radius", 8.0, cast=float)
print(f"[mask-feather][mock] mask={iid} radius={r}")
oid = f"mock-mask-feather-out-{iid}"
emit(mask=oid, width=64, height=64)
