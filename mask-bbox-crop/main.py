"""mask-bbox-crop：按蒙版白色区域 bbox 自动裁剪原图与蒙版，纯客户端 PIL。

流程：mask 二值化(>16)求 bbox → 外扩 padding → 8 对齐 → 同时裁剪 image/mask
→ 等比缩放到长边 target_long_side（边长对齐 8）。
输出缩放后的 crop 图/crop 蒙版，以及原图坐标系的 bbox(x,y,width,height)——
贴回时用 image-upscale 把生成结果缩到 (width,height)，再 image-composite 到 (x,y)。
"""
from flowx_client import emit, emit_result_preview, param, token
from image_io import RESAMPLE, get_pimg, post_pimg


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    padding = param("padding", 64, cast=int)
    target = param("target_long_side", 1536, cast=int)

    im = get_pimg(url, param("image"), tok, mode="RGB")
    mk = get_pimg(url, param("mask"), tok, mode="L")
    if mk.size != im.size:
        mk = mk.resize(im.size, RESAMPLE["bilinear"])

    bbox = mk.point(lambda v: 255 if v > 16 else 0).getbbox()
    if not bbox:
        raise RuntimeError("mask-bbox-crop: mask is empty (no white region)")
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(im.width, x1 + padding)
    y1 = min(im.height, y1 + padding)
    # 8 对齐：原点向下，右/下缘向上（不越界），宽高保证 8 的倍数
    x0 -= x0 % 8
    y0 -= y0 % 8
    x1 = min(im.width, (x1 + 7) // 8 * 8)
    y1 = min(im.height, (y1 + 7) // 8 * 8)
    w, h = x1 - x0, y1 - y0
    w -= w % 8
    h -= h % 8
    if w <= 0 or h <= 0:
        raise RuntimeError(f"mask-bbox-crop: degenerate bbox {w}x{h}")

    crop_im = im.crop((x0, y0, x0 + w, y0 + h))
    crop_mk = mk.crop((x0, y0, x0 + w, y0 + h))

    scale = target / max(w, h)
    cw = max(64, round(w * scale / 8) * 8)
    ch = max(64, round(h * scale / 8) * 8)
    crop_im = crop_im.resize((cw, ch), RESAMPLE["lanczos"])
    crop_mk = crop_mk.resize((cw, ch), RESAMPLE["bilinear"])

    im_id = post_pimg(url, crop_im, tok)
    mk_id = post_pimg(url, crop_mk, tok)
    print(f"[bbox-crop] bbox=({x0},{y0},{w}x{h}) of {im.width}x{im.height} "
          f"pad={padding} -> crop {cw}x{ch} image={im_id} mask={mk_id}", flush=True)
    emit(image=im_id, mask=mk_id, x=x0, y=y0, width=w, height=h,
         crop_width=cw, crop_height=ch)
    emit_result_preview(url, im_id, tok)


if __name__ == "__main__":
    main()
