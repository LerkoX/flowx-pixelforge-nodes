"""mock：不真跑 GPU，输出伪 pose/face 对象 id。"""
from flowx_client import emit, param

vid = param("video", "")
print(f"[wan-animate-preprocess][mock] video={vid} (skipped)", flush=True)
emit(pose_video="mock-pose-video", face_video=vid or "mock-face-video",
     num_frames=81, fps=16)
