"""image-flip mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("image", "mock-image")
mode = param("mode", "horizontal")
print(f"[flip][mock] image={iid} mode={mode}")
oid = f"mock-image-flip-out-{iid}"
emit(image=oid, width=64, height=64)
