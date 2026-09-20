"""sd3-txt2img mock：不调推理服务，回显参数并产出假 IMAGE 引用。"""
from flowx_client import emit, param

prompt = param("prompt", "mock prompt")
w = param("width", 512, int)
h = param("height", 512, int)
steps = param("steps", 28, int)
print(f"[sd3][mock] {w}x{h} steps={steps} prompt={prompt[:40]!r} (no inference)")
emit(image="mock-sd3-image", seed="12345")
