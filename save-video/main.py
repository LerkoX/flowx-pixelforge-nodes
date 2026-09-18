"""save-video：从推理服务下载视频 mp4 保存到本地目录（对标 save-image）。

画布组件经 Studio /api/v1/media/file 资源接口直读本地文件播放（无体积上限、
支持 Range 拖动进度条），不再输出 base64——文件须落在 server 媒体白名单目录
（默认 ~/flowx-output）内，否则画布仅展示保存路径。
"""
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

    emit(file_path=path, size_bytes=len(data))


if __name__ == "__main__":
    main()
