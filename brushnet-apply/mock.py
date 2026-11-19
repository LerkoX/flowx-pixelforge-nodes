"""brushnet-apply mock：模拟条件打包，输出伪 control_id。"""
from flowx_client import emit, param

scale = param("scale", 1.0, cast=float)
print(f"[brushnet-apply][mock] scale={scale} (encode skipped)")
emit(control="mock-brushnet-cond")
