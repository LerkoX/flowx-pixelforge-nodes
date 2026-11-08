"""load-audio：读取本地音频上传到推理服务（POST /audios），输出 AUDIO 对象 ID。

服务端用 ffmpeg 解码重采样为 32kHz 立体声 f32（H3 audio VAE 采样率），
供 minimax-h3-gen 的 references（ref2va 音频参考：嗓音/BGM/环境音锚定）。
对称于 load-image / load-video。
"""
import os
import time
import urllib.error

from flowx_client import emit, param, post_bytes, token

_MAX_BYTES = 64 * 1024 * 1024  # 与服务端 MAX_AUDIO_UPLOAD_MB 默认对齐


def _upload_with_retry(url, data, tok, ctype, fname, retries=4):
    last = None
    for attempt in range(1, retries + 1):
        try:
            return post_bytes(url, f"/audios?filename={fname}", data, tok,
                              timeout=300, content_type=ctype)
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
    path = os.path.expanduser(param("audio_path"))
    tok = token()

    if not os.path.isfile(path):
        raise RuntimeError(f"audio file not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) > _MAX_BYTES:
        raise RuntimeError(f"audio too large: {len(data)} bytes > {_MAX_BYTES}")

    import urllib.parse
    fname = urllib.parse.quote(os.path.basename(path))
    resp = _upload_with_retry(url, data, tok, "application/octet-stream", fname)
    print(f"[upload] {path} ({len(data)} bytes) -> audio={resp['id']} "
          f"{resp.get('seconds')}s@{resp.get('sample_rate')}Hz", flush=True)
    emit(audio=resp["id"], seconds=resp.get("seconds") or 0.0,
         sample_rate=resp.get("sample_rate") or 0)


if __name__ == "__main__":
    main()
