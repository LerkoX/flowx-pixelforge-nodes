"""mask-feather：蒙版高斯羽化（对标 ComfyUI Feather Mask）。"""
from PIL import ImageFilter

from flowx_client import emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg


def process(mask, radius=8.0):
    m = mask.convert("L")
    if radius <= 0:
        return m
    return m.filter(ImageFilter.GaussianBlur(radius))


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    radius = param("radius", 8.0, cast=float)

    m = get_pimg(url, param("mask"), tok, mode="L")
    out = process(m, radius)
    oid = post_pimg(url, out, tok)
    print(f"[mask-feather] mask={param('mask')} radius={radius} -> mask={oid}",
          flush=True)
    emit(mask=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
