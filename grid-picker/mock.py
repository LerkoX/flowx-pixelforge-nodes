"""grid-picker mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[grid-picker][mock] files=%s crops=%s cell=%s"
      % (param("files", "[]")[:40], param("crops", "")[:40], param("cell_size", "640")))
emit(count=9, images='["mock-c1","mock-c2"]',
     **{f"img{i}": f"mock-cell-{i}" for i in range(1, 10)})
