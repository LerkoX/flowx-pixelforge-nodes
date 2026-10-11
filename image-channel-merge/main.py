"""image-channel-merge：通道合并（对标 ComfyUI），纯节点侧 PIL。"""
from flowx_client import emit, emit_result_preview, param, token
from image_filters import channel_merge
from image_io import get_pimg, post_pimg


def process(r_image, g_image=None, b_image=None, alpha_image=None):
    return channel_merge(r_image, g_image, b_image, alpha_image)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    r_img = get_pimg(url, param("r_image"), tok, mode="L")
    g_img = get_pimg(url, param("g_image"), tok, mode="L")
    b_img = get_pimg(url, param("b_image"), tok, mode="L")
    a_id = param("alpha_image", "")
    a_img = get_pimg(url, a_id, tok, mode="L") if a_id else None
    out = process(r_img, g_img, b_img, a_img)
    oid = post_pimg(url, out, tok)
    print(f"[image-channel-merge] r={param('r_image')} g={param('g_image')} "
          f"b={param('b_image')} a={a_id or '-'} -> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
