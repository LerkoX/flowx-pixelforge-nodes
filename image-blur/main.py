"""image-blur：模糊（对标 ComfyUI），纯节点侧 PIL。"""
from flowx_client import emit, emit_result_preview, param, token
from image_filters import blur
from image_io import get_pimg, post_pimg


def process(im, **kw):
    return blur(im, **kw)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    im = get_pimg(url, param("image"), tok, mode="RGB")
    out = blur(im, mode=param("mode", "gaussian"), radius=param("radius", 4.0, cast=float), angle=param("angle", 0.0, cast=float))
    oid = post_pimg(url, out, tok)
    print(f"[image-blur] image={param('image')} -> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
