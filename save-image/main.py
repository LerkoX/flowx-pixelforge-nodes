"""save-image：从推理服务下载图像 PNG 保存到本地，并取缩略图供画布预览。"""
import base64
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

    # 缩略图（JPEG ~256px，base64 供画布组件内嵌预览）；失败不阻断主流程
    thumb_b64 = ""
    try:
        thumb = get_bytes(url, f"/images/{image_id}?index={index}&thumb=256",
                          tok, timeout=60)
        thumb_b64 = base64.b64encode(thumb).decode()
        print(f"[save] thumbnail {len(thumb)} bytes", flush=True)
    except RuntimeError as e:
        print(f"[save] thumbnail skipped: {e}", flush=True)

    # 弹窗大图（JPEG ~1024px，base64 供画布组件点击放大预览）；失败不阻断
    preview_b64 = ""
    try:
        preview = get_bytes(url, f"/images/{image_id}?index={index}&thumb=1024",
                            tok, timeout=60)
        preview_b64 = base64.b64encode(preview).decode()
        print(f"[save] preview {len(preview)} bytes", flush=True)
    except RuntimeError as e:
        print(f"[save] preview skipped: {e}", flush=True)

    emit(file_path=path, size_bytes=len(data), thumbnail_b64=thumb_b64,
         preview_b64=preview_b64)


if __name__ == "__main__":
    main()
