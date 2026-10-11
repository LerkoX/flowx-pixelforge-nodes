"""matting-composite：抠图换底合成（纯节点侧 PIL，不占显存）。

bg_mode=color（纯色底，默认白底）或 image（背景图 cover 铺满）；
前景取 image 自带 alpha，或 mask 参数（灰度）覆盖；支持羽化与偏移。
"""
from flowx_client import emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg


def _parse_color(s):
    """#RGB / #RRGGBB / #RRGGBBAA → (r,g,b,a)。"""
    s = (s or "#FFFFFF").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) == 6:
        s += "FF"
    if len(s) != 8:
        raise ValueError(f"bg_color 非法：{s!r}（用 #RRGGBB 或 #RRGGBBAA）")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4, 6))


def _cover(img, size):
    """按 cover 语义缩放并居中裁切到 size。"""
    from PIL import Image
    tw, th = size
    sw, sh = img.size
    k = max(tw / sw, th / sh)
    nw, nh = max(1, round(sw * k)), max(1, round(sh * k))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def process(fg, mask=None, bg_mode="color", bg_color="#FFFFFF", bg_img=None,
            feather=0, x=0, y=0):
    """合成：fg(RGB/RGBA) + 可选 mask(L) → RGB 合成图（画布尺寸 = 底色尺寸）。"""
    from PIL import Image, ImageFilter

    fg = fg.convert("RGBA")
    if mask is not None:
        m = mask.convert("L")
        if m.size != fg.size:
            m = m.resize(fg.size, Image.BILINEAR)
        if float(feather) > 0:
            m = m.filter(ImageFilter.GaussianBlur(float(feather)))
        fg.putalpha(m)
    elif float(feather) > 0:
        a = fg.getchannel("A").filter(ImageFilter.GaussianBlur(float(feather)))
        fg.putalpha(a)

    if bg_mode == "image":
        if bg_img is None:
            raise ValueError("bg_mode=image 需要 bg_image（背景图对象 ID）")
        canvas = _cover(bg_img.convert("RGB"), fg.size).convert("RGBA")
    else:
        canvas = Image.new("RGBA", fg.size, _parse_color(bg_color))
    canvas.alpha_composite(fg, (int(x), int(y)))
    return canvas.convert("RGB")


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    fg = get_pimg(url, param("image"), tok, mode="RGBA")
    mask_id = param("mask", "")
    mask = get_pimg(url, mask_id, tok, mode="L") if mask_id else None
    bg_mode = param("bg_mode", "color")
    bg_img = None
    if bg_mode == "image":
        bg_id = param("bg_image", "")
        if not bg_id:
            raise ValueError("matting-composite: bg_mode=image 需要 bg_image")
        bg_img = get_pimg(url, bg_id, tok, mode="RGB")
    out = process(fg, mask=mask, bg_mode=bg_mode, bg_color=param("bg_color", "#FFFFFF"),
                  bg_img=bg_img, feather=param("feather", 0, cast=float),
                  x=param("x", 0, cast=int), y=param("y", 0, cast=int))
    oid = post_pimg(url, out, tok)
    print(f"[matting-composite] image={param('image')} mask={mask_id or '-'} "
          f"bg={bg_mode} -> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
