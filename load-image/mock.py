"""load-image mock：不读文件，直接生成伪 image_id。"""
import os

from flowx_client import emit, param

path = param("image_path", "mock.png")
iid = f"mock-image-{os.path.basename(path)}"
print(f"[upload][mock] {path} -> {iid}")
emit(image=iid)
