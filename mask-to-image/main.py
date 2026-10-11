"""mask-to-image：蒙版转可视化灰度图像（对标 ComfyUI Mask To Image）。"""
from flowx_client import emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg


def process(mask):
    return mask.convert("L").convert("RGB")


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    m = get_pimg(url, param("mask"), tok, mode="L")
    out = process(m)
    oid = post_pimg(url, out, tok)
    print(f"[mask-to-image] mask={param('mask')} -> image={oid} "
          f"{out.width}x{out.height}", flush=True)
    emit(image=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
