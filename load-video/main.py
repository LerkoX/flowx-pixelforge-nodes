"""load-video：读取 FlowX 侧本地视频，上传到推理服务并登记为 VIDEO 对象。

两步（对齐 load-image 的本地图路径）：
1. POST /videos 上传原片字节 —— 服务端落盘 VIDEO_INPUT_DIR 并回传服务端路径
   （main.py 只做字节入站，解码由算子负责）；
2. 提交 video.load 算子把该文件登记成 VIDEO 对象：不给归一化参数时**原字节直通**
   （拷进对象目录，不重编码，无损）；给了 start/end/stride/max_frames/fps/
   width/height 才解码归一化后重编码。

输出 video 对象 ID + fps/帧数/宽高，供 video-repaint / video-v2v /
video-to-frames / save-video 消费。
"""
import os
import time
import urllib.error

from executor_base import run_op
from flowx_client import param, post_bytes, token

# 与推理服务 MAX_VIDEO_UPLOAD_MB 默认值（512）对齐：超了不一定失败（服务端可调大），
# 但先提示，避免隧道上白等几分钟才拿到 413
_WARN_BYTES = 512 * 1024 * 1024


def _upload(url, path, tok, retries=5):
    """上传原片；免费隧道会抖动返回 HTML 错误页/连接重置，重试几次。

    视频体积大、传输久，超时给足（1800s）——中途断开只会重传，服务端落盘名带
    时间戳前缀，重传不会覆盖或串味。"""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) > _WARN_BYTES:
        print(f"[upload] 警告：{len(data)} bytes 超过服务端默认上限"
              f"（MAX_VIDEO_UPLOAD_MB=512）；若 413 请调大该环境变量", flush=True)
    ctype = ("video/mp4" if path.lower().endswith((".mp4", ".m4v"))
             else "application/octet-stream")
    last = None
    for attempt in range(1, retries + 1):
        try:
            return post_bytes(url, "/videos?filename=" + os.path.basename(path),
                              data, tok, timeout=1800, content_type=ctype)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                OSError) as e:
            last = e
            print(f"[upload] attempt {attempt}/{retries} failed: {e}", flush=True)
            if attempt < retries:
                time.sleep(min(attempt * 3, 10))
    raise RuntimeError(f"video upload failed after {retries} attempts: {last}")


def main():
    url = param("service_url").rstrip("/")
    path = os.path.expanduser(param("video_path"))
    tok = token()

    if not os.path.isfile(path):
        raise RuntimeError(f"video file not found: {path}")
    resp = _upload(url, path, tok)
    uploaded = resp.get("path")
    if not uploaded:
        raise RuntimeError(
            f"上传未返回服务端路径（推理服务版本过旧？POST /videos 应答：{resp}）")
    print(f"[load-video] {path} -> {uploaded} ({resp.get('bytes')} bytes)",
          flush=True)

    # 服务端算子自注册 + 异步执行（video.load 走 /jobs：长视频解码是分钟级）
    run_op("video.load", {
        "path": uploaded,
        "start": param("start", 0, int),
        "end": param("end", 0, int),
        "stride": param("stride", 1, int),
        "max_frames": param("max_frames", 0, int),
        "fps": param("fps", 0, float),
        "width": param("width", 0, int),
        "height": param("height", 0, int),
    }, emit_keys=["video", "fps", "num_frames", "width", "height"])


if __name__ == "__main__":
    main()
