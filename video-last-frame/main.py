"""video-last-frame：视频末帧抽取——长视频末帧续接的闭环件。

下载推理服务视频对象 → 本地 ffmpeg 抽倒数帧 → POST /images 上传为 IMAGE 对象，
输出直接可接 H3Gen.image（下一段的首帧条件）。纯本地节点（ffmpeg 在 Studio
机器执行），不消耗推理服务 GPU。
"""
import os
import subprocess
import tempfile
import time
import urllib.error

from flowx_client import (emit, emit_result_preview, get_bytes, param,
                          post_bytes, token)


def _upload_with_retry(url, data, tok, ctype, retries=4):
    last = None
    for attempt in range(1, retries + 1):
        try:
            return post_bytes(url, "/images", data, tok, timeout=300,
                              content_type=ctype)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError) as e:
            last = e
            print(f"[upload] attempt {attempt}/{retries} failed: {e}",
                  flush=True)
            if attempt < retries:
                time.sleep(min(attempt * 3, 10))
    raise RuntimeError(f"upload failed after {retries} attempts: {last}")


def main():
    url = param("service_url").rstrip("/")
    video_id = param("video")
    offset = param("offset", 0.2, float)
    tok = token()

    data = get_bytes(url, f"/videos/{video_id}", tok, timeout=600)
    print(f"[last-frame] video={video_id} downloaded {len(data)} bytes",
          flush=True)

    with tempfile.TemporaryDirectory(prefix="flowx-lf-") as td:
        src = os.path.join(td, "src.mp4")
        frame = os.path.join(td, "last.jpg")
        with open(src, "wb") as f:
            f.write(data)
        # -sseof 从末尾回退抽帧（避开可能的黑场/渐隐尾帧）；
        # 某些封装 -sseof 不准时逐级加大回退重试（不用 reverse 滤镜——
        # 它要把整段解码进内存，手机 Termux 上吃不消）
        ok = False
        for off in (offset, 0.5, 1.0):
            r = subprocess.run(
                ["ffmpeg", "-y", "-sseof", f"-{off}", "-i", src,
                 "-frames:v", "1", "-q:v", "2", frame],
                capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and os.path.isfile(frame) \
                    and os.path.getsize(frame) > 0:
                ok = True
                break
        if not ok:
            raise RuntimeError(f"ffmpeg extract failed: {r.stderr[-300:]}")
        with open(frame, "rb") as f:
            jpg = f.read()

    resp = _upload_with_retry(url, jpg, tok, "image/jpeg")
    print(f"[last-frame] uploaded -> image={resp['id']} "
          f"{resp.get('width')}x{resp.get('height')}", flush=True)
    emit(image=resp["id"], width=resp.get("width") or 0,
         height=resp.get("height") or 0)
    # 画布预览抽到的末帧（确认续接起点）
    emit_result_preview(url, resp["id"], tok)


if __name__ == "__main__":
    main()
