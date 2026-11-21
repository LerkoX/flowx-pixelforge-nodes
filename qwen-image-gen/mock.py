"""qwen-image-gen mock：不真跑 GPU，输出伪 image id（对齐真实输出形态）。"""
import time

from flowx_client import emit, param

w = param("width", 1328, int)
h = param("height", 1328, int)
print(f"[qwen-image-gen][mock] {w}x{h} steps={param('steps', 50)}", flush=True)
time.sleep(0.2)
print("[qwen-image-gen][mock] generation skipped", flush=True)
emit(image="mock-qwen-image", seed=42)
