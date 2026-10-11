"""image-channel-extract：通道分离（对标 ComfyUI），纯节点侧 PIL。"""
from flowx_client import emit, emit_result_preview, param, token
from image_filters import channel_extract
from image_io import get_pimg, post_pimg


def process(im, **kw):
    return channel_extract(im, **kw)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    im = get_pimg(url, param("image"), tok, mode=None)
    out = channel_extract(im, channel=param("channel", "luminance"))
    oid = post_pimg(url, out, tok)
    print(f"[image-channel-extract] image={param('image')} -> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
