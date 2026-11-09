"""wan-gen mock：不真跑 GPU，输出伪 video id（对齐真实输出形态）。"""
import time

from flowx_client import emit, param

frames = param("num_frames", 81, int)
w = param("width", 832, int)
h = param("height", 480, int)
print(f"[wan-gen][mock] frames={frames} {w}x{h} steps={param('steps', 30)}",
      flush=True)
time.sleep(0.2)
print("[wan-gen][mock] generation skipped", flush=True)
emit(video="mock-wan-video", seed=42, fps=16)
