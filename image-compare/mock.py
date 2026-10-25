"""image-compare mock：不访问服务，透传伪对象 id，验证参数注入与输出提取。"""
import json

from flowx_client import emit, param

_before = param("image_before", "mock-before")
_after = param("image_after", "mock-after")
print(f"[image-compare][mock] before={_before} after={_after}")
emit(before=_before, after=_after,
     __inputs_resolved=json.dumps({"image_before": _before, "image_after": _after},
                                  ensure_ascii=False))
