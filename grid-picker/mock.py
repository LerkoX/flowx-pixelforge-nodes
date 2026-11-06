"""grid-picker mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[grid-picker][mock] files=%s crops=%s cell=%s"
      % (param("files", "[]")[:40], param("crops", "")[:40], param("cell_size", "640")))
emit(image="mock-grid-picker", width=1960, height=2600, count=9)
