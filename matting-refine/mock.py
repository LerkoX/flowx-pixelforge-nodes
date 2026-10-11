"""matting-refine mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_mask = param("mask", "mock-mask")
print("[matting-refine][mock] mask=%s threshold=%s feather=%s"
      % (_mask, param("threshold", "0"), param("feather", "0")))
emit(mask="mock-mask-refined-" + str(_mask), width=64, height=64)
