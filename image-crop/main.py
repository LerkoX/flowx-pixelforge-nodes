"""image-crop：矩形裁剪（对图像与蒙版通用——蒙版即灰度图像），纯客户端 PIL。"""
from flowx_client import emit, param, token
from image_io import get_pimg, post_pimg, thumb_data_url


def process(im, x, y, width, height):
    if width <= 0 or height <= 0:
        raise RuntimeError(f"width/height must be > 0, got {width}x{height}")
    if x < 0 or y < 0:
        raise RuntimeError(f"x/y must be >= 0, got ({x},{y})")
    if x + width > im.width or y + height > im.height:
        raise RuntimeError(
            f"crop box ({x},{y},{width}x{height}) exceeds image {im.width}x{im.height}")
    return im.crop((x, y, x + width, y + height))


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    x = param("x", 0, cast=int)
    y = param("y", 0, cast=int)
    w = param("width", 0, cast=int)
    h = param("height", 0, cast=int)

    im = get_pimg(url, param("image"), tok)
    out = process(im, x, y, w, h)
    oid = post_pimg(url, out, tok)
    print(f"[crop] image={param('image')} ({x},{y},{w}x{h}) of {im.width}x{im.height} "
          f"-> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height,
         thumb_b64=thumb_data_url(out))


if __name__ == "__main__":
    main()
