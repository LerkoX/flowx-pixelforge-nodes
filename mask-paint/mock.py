"""mask-paint mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("image", "mock-image")
raw = param("strokes_json", "")
print(f"[mask-paint][mock] image={iid} strokes_json={len(raw)} chars")
emit(mask=f"mock-mask-paint-out-{iid}", width=64, height=64, coverage=0.1)
