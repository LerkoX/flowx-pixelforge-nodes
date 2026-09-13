"""inference-ensure：推理服务健康检查，等待就绪后输出 service_url。"""
import sys
import time

from flowx_client import emit, get_json, param, token


def main():
    url = param("service_url").rstrip("/")
    wait = param("wait_seconds", 60, int)
    tok = token()

    deadline = time.time() + wait
    attempt = 0
    while True:
        attempt += 1
        try:
            info = get_json(url, "/health", tok, timeout=10)
            break
        except RuntimeError as e:
            if time.time() >= deadline:
                print(f"[ensure] service not ready after {wait}s: {e}", file=sys.stderr)
                raise
            print(f"[ensure] attempt {attempt}: not ready ({e}), retry in 3s ...", flush=True)
            time.sleep(3)

    gpu = info.get("gpu_name", "unknown-gpu")
    print(f"[ensure] service ready: {url} gpu={gpu} "
          f"vram={info.get('vram_free_mb', '?')}/{info.get('vram_total_mb', '?')}MB "
          f"resident={info.get('resident_models', [])}", flush=True)
    emit(service_url=url, status=info.get("status", "ok"), gpu_name=gpu)


if __name__ == "__main__":
    main()
