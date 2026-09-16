"""save-video：从推理服务下载视频 mp4 保存到本地目录（对标 save-image）。

小体积视频（<= embed_max_mb）内嵌 base64 输出，供画布组件 <video> 内嵌播放；
大文件跳过内嵌，仅用 file_path。
"""
import base64
import os
import re
import time

from flowx_client import emit, get_bytes, param, token


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def main():
    url = param("service_url").rstrip("/")
    video_id = param("video")
    prefix = sanitize(param("filename_prefix", "flowx-video"))
    out_dir = os.path.expanduser(param("output_dir", "~/flowx-output"))
    tok = token()

    data = get_bytes(url, f"/videos/{video_id}", tok, timeout=600)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    with open(path, "wb") as f:
        f.write(data)
    print(f"[save] video={video_id} -> {path} ({len(data)} bytes)", flush=True)

    # 小体积内嵌 base64 供画布 <video> 播放；失败/超限不阻断主流程
    video_b64 = ""
    limit = param("embed_max_mb", 4, int) * 1024 * 1024
    if len(data) <= limit:
        video_b64 = base64.b64encode(data).decode()
    else:
        print(f"[save] embed skipped: {len(data)} bytes > {limit}", flush=True)

    emit(file_path=path, size_bytes=len(data), video_b64=video_b64)


if __name__ == "__main__":
    main()
