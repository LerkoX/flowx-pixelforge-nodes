"""minimax-h3-gen mock：模拟生成耗时，输出伪 video_id。"""
import random
import time

from flowx_client import emit, param

seed = param("seed", -1, int)
steps = param("steps", 4, int)
frames = param("num_frames", 125, int)
if seed < 0:
    seed = random.randint(0, 2**32 - 1)
for i in range(steps):
    print(f"[minimax-h3-gen][mock] denoise {i + 1}/{steps}", flush=True)
    time.sleep(0.1)
print(f"[minimax-h3-gen][mock] frames={frames} seed={seed} "
      f"(video+audio skipped)", flush=True)
emit(video=f"mock-h3-video-{seed}", seed=seed)
