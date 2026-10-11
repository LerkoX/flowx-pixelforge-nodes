"""mask-from-image mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("image", "mock-image")
ch = param("channel", "luminance")
print(f"[mask-from-image][mock] image={iid} channel={ch}")
oid = f"mock-mask-from-image-out-{iid}"
emit(mask=oid, width=64, height=64)
