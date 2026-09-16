"""video-gen mock：模拟异步任务耗时，输出伪 video_id。"""
import random
import time

from flowx_client import emit, param

seed = param("seed", -1, int)
steps = param("steps", 50, int)
frames = param("num_frames", 121, int)
if seed < 0:
    seed = random.randint(0, 2**32 - 1)
for i in range(0, steps, 10):  # 模拟进度日志
    print(f"[video-gen][mock] running {min(i + 10, steps)}/{steps}", flush=True)
    time.sleep(0.1)
print(f"[video-gen][mock] frames={frames} seed={seed} (sampling skipped)")
emit(video=f"mock-video-{seed}", seed=seed)
