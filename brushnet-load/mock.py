"""brushnet-load mock：模拟加载，输出伪 control_net_id。"""
from flowx_client import emit, param

name = param("name", "brushnet_random_mask")
print(f"[brushnet-load][mock] name={name} (load skipped)")
emit(control_net=f"mock-brushnet-{name}")
