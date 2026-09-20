"""mask-invert：蒙版反相（对标 ComfyUI Invert Mask）。"""
from PIL import ImageOps

from flowx_client import emit, param, token
from image_io import get_pimg, post_pimg, thumb_data_url


def process(mask):
    return ImageOps.invert(mask.convert("L"))


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    m = get_pimg(url, param("mask"), tok, mode="L")
    out = process(m)
    oid = post_pimg(url, out, tok)
    print(f"[mask-invert] mask={param('mask')} -> mask={oid}", flush=True)
    emit(mask=oid, width=out.width, height=out.height,
         thumb_b64=thumb_data_url(out))


if __name__ == "__main__":
    main()
