"""load-image：读取本地图片上传到推理服务（POST /images），输出图像对象 ID。"""
import os

from flowx_client import emit, param, post_bytes, token

_MAX_BYTES = 32 * 1024 * 1024  # 与服务端 MAX_UPLOAD_MB 默认值对齐


def main():
    url = param("service_url").rstrip("/")
    path = os.path.expanduser(param("image_path"))
    tok = token()

    if not os.path.isfile(path):
        raise RuntimeError(f"image file not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) > _MAX_BYTES:
        raise RuntimeError(f"image too large: {len(data)} bytes > {_MAX_BYTES}")

    ctype = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    resp = post_bytes(url, "/images", data, tok, timeout=300, content_type=ctype)
    print(f"[upload] {path} ({len(data)} bytes) -> image={resp['id']} "
          f"{resp.get('width')}x{resp.get('height')}", flush=True)
    emit(image=resp["id"])


if __name__ == "__main__":
    main()
