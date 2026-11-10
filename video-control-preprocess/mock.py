"""mock。"""
from flowx_client import emit, param
print(f"[video-control-preprocess][mock] mode={param('mode','depth')}", flush=True)
emit(video="mock-control-video", num_frames=81, fps=16)
