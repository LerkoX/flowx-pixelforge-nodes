"""save-mesh mock：不落盘，只回报伪路径。"""
from flowx_client import emit, param

mesh = param("mesh", "mock")
path = f"~/flowx-output/mock-mesh-from-{mesh}.glb"
print(f"[save-mesh][mock] mesh={mesh} -> {path}")
emit(file_path=path, size_bytes=0)
