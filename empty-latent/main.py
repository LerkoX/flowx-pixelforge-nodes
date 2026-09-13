"""empty-latent：Empty Latent Image，按尺寸在服务端创建零 latent。"""
from flowx_client import call_op, emit, param, token


def main():
    url = param("service_url").rstrip("/")
    width = param("width", 512, int)
    height = param("height", 512, int)
    batch = param("batch_size", 1, int)
    tok = token()

    out = call_op(url, "latent.empty",
                  {"width": width, "height": height, "batch_size": batch},
                  tok, timeout=120)
    print(f"[latent] {width}x{height} batch={batch} -> latent={out['latent']}", flush=True)
    emit(latent=out["latent"])


if __name__ == "__main__":
    main()
