"""save-image：从推理服务下载图像 PNG 保存到本地。

画布组件经 Studio /api/v1/media/file 资源接口直读本地文件展示，不再输出
base64 缩略图——文件须落在 server 媒体白名单目录（默认 ~/flowx-output）内。
"""
import os
import re
import time

from flowx_client import emit, get_bytes, param, token


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def main():
    url = param("service_url").rstrip("/")
    image_id = param("image")
    prefix = sanitize(param("filename_prefix", "flowx"))
    out_dir = os.path.expanduser(param("output_dir", "~/flowx-output"))
    index = param("index", 0, int)
    tok = token()

    data = get_bytes(url, f"/images/{image_id}?index={index}", tok, timeout=300)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.png")
    with open(path, "wb") as f:
        f.write(data)
    print(f"[save] image={image_id} -> {path} ({len(data)} bytes)", flush=True)

    emit(file_path=path, size_bytes=len(data))


if __name__ == "__main__":
    main()
