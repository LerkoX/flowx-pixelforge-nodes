"""mask-to-image mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("mask", "mock-mask")
print(f"[mask-to-image][mock] mask={iid}")
oid = f"mock-mask-to-image-out-{iid}"
emit(image=oid, width=64, height=64)
