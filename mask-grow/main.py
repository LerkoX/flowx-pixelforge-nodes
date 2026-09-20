"""mask-grow：蒙版扩张/收缩（对标 ComfyUI Grow Mask）。

radius > 0 扩张白色区域（MaxFilter），< 0 收缩（MinFilter），0 原样返回。
滤波核 = 2*|radius|+1（奇数），|radius| 即近似像素半径。
"""
from PIL import ImageFilter

from flowx_client import emit, param, token
from image_io import get_pimg, post_pimg, thumb_data_url


def process(mask, radius=0):
    m = mask.convert("L")
    if radius == 0:
        return m
    size = 2 * abs(radius) + 1
    f = ImageFilter.MaxFilter(size) if radius > 0 else ImageFilter.MinFilter(size)
    return m.filter(f)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    radius = param("radius", 0, cast=int)

    m = get_pimg(url, param("mask"), tok, mode="L")
    out = process(m, radius)
    oid = post_pimg(url, out, tok)
    print(f"[mask-grow] mask={param('mask')} radius={radius} -> mask={oid}",
          flush=True)
    emit(mask=oid, width=out.width, height=out.height,
         thumb_b64=thumb_data_url(out))


if __name__ == "__main__":
    main()
