"""model-unload mock：不调服务，返回伪卸载结果。"""
from flowx_client import emit, param

target = param("target", "")
print(f"[model-unload][mock] target='{target or '(all)'}'")
emit(unloaded="mock-model", resident="(empty)")
