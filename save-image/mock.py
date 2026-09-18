"""save-image mock：生成 1x1 伪 PNG 并保存，验证参数注入与输出提取。"""
import base64
import os
import time

from flowx_client import emit, param

# 1x1 透明 PNG
PNG_1PX = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

image_id = param("image", "mock")
prefix = param("filename_prefix", "flowx")
out_dir = os.path.expanduser(param("output_dir", "~/flowx-output"))
os.makedirs(out_dir, exist_ok=True)
path = os.path.join(out_dir, f"{prefix}_mock_{int(time.time())}.png")
with open(path, "wb") as f:
    f.write(PNG_1PX)
print(f"[save][mock] image={image_id} -> {path} (1x1 placeholder)")
emit(file_path=path, size_bytes=len(PNG_1PX))
