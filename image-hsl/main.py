"""image-hsl：色相饱和度明度（对标 ComfyUI），纯节点侧 PIL。"""
from flowx_client import emit, emit_result_preview, param, token
from image_filters import hsl
from image_io import get_pimg, post_pimg


def process(im, **kw):
    return hsl(im, **kw)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    im = get_pimg(url, param("image"), tok, mode="RGB")
    out = hsl(im, hue=param("hue", 0.0, cast=float), saturation=param("saturation", 1.0, cast=float), lightness=param("lightness", 0.0, cast=float))
    oid = post_pimg(url, out, tok)
    print(f"[image-hsl] image={param('image')} -> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
