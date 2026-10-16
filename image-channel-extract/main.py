"""image-channel-extract：通道分离（对标 ComfyUI），纯节点侧 PIL。"""
from flowx_client import emit_input_media, emit, emit_result_preview, param, token
from image_filters import channel_extract
from image_io import get_pimg, post_pimg


def process(im, **kw):
    return channel_extract(im, **kw)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    # 输入原图物化：画布对比控件的「原图」侧（推理服务对象 GC/TTL 后
    # 必然 404，必须在执行当时落一份到 Studio 媒体缓存）
    emit_input_media(url, param("image"), tok)
    im = get_pimg(url, param("image"), tok, mode=None)
    out = channel_extract(im, channel=param("channel", "luminance"))
    oid = post_pimg(url, out, tok)
    print(f"[image-channel-extract] image={param('image')} -> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
