"""load-image：读取本地图片上传到推理服务（POST /images），输出图像对象 ID。"""
import os
import time
import urllib.error

from flowx_client import (emit, emit_result_preview, ensure_plugin, param,
                          post_bytes, token)

_MAX_BYTES = 32 * 1024 * 1024  # 与服务端 MAX_UPLOAD_MB 默认值对齐


def _upload_with_retry(url, data, tok, ctype, retries=5):
    """cpolar 免费隧道会抖动返回 HTML 错误页（4xx/5xx），上传重试几次。"""
    last = None
    for attempt in range(1, retries + 1):
        try:
            return post_bytes(url, "/images", data, tok, timeout=300, content_type=ctype)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last = e
            print(f"[upload] attempt {attempt}/{retries} failed: {e}", flush=True)
            if attempt < retries:
                time.sleep(min(attempt * 3, 10))
    raise RuntimeError(f"upload failed after {retries} attempts: {last}")


def main():
    url = param("service_url").rstrip("/")
    path = os.path.expanduser(param("image_path"))
    tok = token()
    # image.load 算子（服务端 INPUT_DIR 本地图）随节点自注册；
    # 本节点主路径走 POST /images 上传客户端本地图

    if not os.path.isfile(path):
        raise RuntimeError(f"image file not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) > _MAX_BYTES:
        raise RuntimeError(f"image too large: {len(data)} bytes > {_MAX_BYTES}")

    # 插件注册同样过隧道，抖动时重试（与上传同策略）
    last = None
    for attempt in range(1, 4):
        try:
            ensure_plugin(url, "image.load", tok=tok)
            break
        except Exception as e:  # noqa: BLE001 - 隧道抖动形态多样，统一重试
            last = e
            print(f"[plugin] ensure attempt {attempt}/3 failed: {e}", flush=True)
            time.sleep(min(attempt * 3, 10))
    else:
        raise RuntimeError(f"ensure_plugin failed after 3 attempts: {last}")

    ctype = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    resp = _upload_with_retry(url, data, tok, ctype)
    print(f"[upload] {path} ({len(data)} bytes) -> image={resp['id']} "
          f"{resp.get('width')}x{resp.get('height')}", flush=True)
    emit(image=resp["id"], width=resp.get("width") or 0,
         height=resp.get("height") or 0)
    # 结果预览帧：画布立刻显示"到底加载了哪张图"（wf157 反馈：加载节点看不到图）。
    # host_base() 优先 SERVICE_URL_HOST（docker 节点容器内不可达时由 workflow 绑 Param.service_url）
    emit_result_preview(url, resp["id"], tok)


if __name__ == "__main__":
    main()
