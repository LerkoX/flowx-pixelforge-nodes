"""save-video mock：写入伪 mp4 字节并保存，验证参数注入与输出提取。"""
import os
import time

from flowx_client import emit, param

# ftyp box 开头的伪 mp4（仅供打通链路，不可播放）
FAKE_MP4 = (b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom"
            b"mock-video-bytes")

video_id = param("video", "mock")
prefix = param("filename_prefix", "flowx-video")
out_dir = os.path.expanduser(param("output_dir", "~/flowx-output"))
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, f"{prefix}_mock_{int(time.time())}.mp4")
with open(path, "wb") as f:
    f.write(FAKE_MP4)
print(f"[save][mock] video={video_id} -> {path} (fake mp4 placeholder)")
emit(file_path=path, size_bytes=len(FAKE_MP4))
