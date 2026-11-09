"""wan-animate mock：不真跑 GPU，输出伪 video id。"""
import time

from flowx_client import emit, param

mode = param("mode", "animate")
print(f"[wan-animate][mock] mode={mode} "
      f"seg={param('segment_frame_length', 77)}", flush=True)
time.sleep(0.2)
print("[wan-animate][mock] animation skipped", flush=True)
emit(video="mock-wan-animate-video", seed=42, fps=30)
