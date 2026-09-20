"""image-flip：水平/垂直镜像翻转（对标 ComfyUI Image Flip），纯客户端 PIL。"""
from PIL import ImageOps

from flowx_client import emit, param, token
from image_io import get_pimg, post_pimg, thumb_data_url


def process(im, mode="horizontal"):
    return ImageOps.mirror(im) if mode == "horizontal" else ImageOps.flip(im)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    mode = param("mode", "horizontal")
    if mode not in ("horizontal", "vertical"):
        raise RuntimeError(f"mode must be horizontal|vertical, got {mode!r}")

    im = get_pimg(url, param("image"), tok, mode="RGB")
    out = process(im, mode)
    oid = post_pimg(url, out, tok)
    print(f"[flip] image={param('image')} mode={mode} -> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height,
         thumb_b64=thumb_data_url(out))


if __name__ == "__main__":
    main()
