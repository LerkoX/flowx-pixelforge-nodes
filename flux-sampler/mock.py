"""flux-sampler mock：不调服务，回显参数产出假图像引用与种子。"""
from flowx_client import emit, param

w = param("width", 1024, int)
h = param("height", 1024, int)
steps = param("steps", 4, int)
print(f"[flux-sampler][mock] {w}x{h} steps={steps}")
emit(image="mock-flux-image", seed=42)
