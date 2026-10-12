"""mask-paint mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("image", "")
path = param("image_path", "")
if not iid and path:
    iid = f"mock-upload-{path.rsplit('/', 1)[-1]}"
ref = param("strokes_ref", "")
inline = param("strokes_json", "")
print(f"[mask-paint][mock] image={iid or '(none)'} image_path={path or '-'} "
      f"strokes_ref={ref or '-'} strokes_json={len(inline)} chars")
emit(image=iid, mask=f"mock-mask-paint-out-{iid}",
     width=768, height=1280, coverage=0.1)
