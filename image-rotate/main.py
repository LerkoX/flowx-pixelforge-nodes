"""image-rotate：任意角度旋转（对标 ComfyUI Image Rotate），纯客户端 PIL。"""
from flowx_client import emit, param, token
from image_io import RESAMPLE, get_pimg, pbool, post_pimg, thumb_data_url


def process(im, angle=0.0, expand=True, resample="bicubic"):
    return im.rotate(angle, expand=expand,
                     resample=RESAMPLE.get(resample, RESAMPLE["bicubic"]),
                     fillcolor=(0, 0, 0))


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    angle = param("angle", 0.0, cast=float)
    expand = pbool(param("expand", "true"))
    resample = param("resample", "bicubic")

    im = get_pimg(url, param("image"), tok, mode="RGB")
    out = process(im, angle, expand, resample)
    oid = post_pimg(url, out, tok)
    print(f"[rotate] image={param('image')} angle={angle} expand={expand} "
          f"{im.width}x{im.height} -> image={oid} {out.width}x{out.height}", flush=True)
    emit(image=oid, width=out.width, height=out.height,
         thumb_b64=thumb_data_url(out))


if __name__ == "__main__":
    main()
