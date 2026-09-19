"""image-upscale：图像放大（hires.fix 前置），图像对象引用 → 放大后的图像对象引用。"""
from flowx_client import call_op, emit, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    scale = param("scale", 2.0, cast=float)
    width = param("width", 0, cast=int)
    height = param("height", 0, cast=int)
    method = param("method", "lanczos")

    out = call_op(url, "image.upscale",
                  {"image": ref(param("image")),
                   "scale": scale, "width": width, "height": height,
                   "method": method},
                  tok, timeout=600)
    print(f"[upscale] image={param('image')} scale={scale} "
          f"target={width}x{height} method={method} -> image={out['image']}",
          flush=True)
    emit(image=out["image"])


if __name__ == "__main__":
    main()
