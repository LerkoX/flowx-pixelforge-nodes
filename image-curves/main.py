"""image-curves：曲线映射（对标 ComfyUI），纯节点侧 PIL。"""
from flowx_client import emit, emit_result_preview, param, token
from image_filters import curves
from image_io import get_pimg, post_pimg


def process(im, **kw):
    return curves(im, **kw)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    im = get_pimg(url, param("image"), tok, mode="RGB")
    out = curves(im, channel=param("channel", "all"), points=param("points", "0,0;255,255"))
    oid = post_pimg(url, out, tok)
    print(f"[image-curves] image={param('image')} -> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
