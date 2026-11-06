"""grid-compose：朋友圈九宫格底图（纯节点侧 PIL，不占显存）。

9 张图 → 3×3 正方形网格底图：每格 cover 裁方形，可调间隙/外边距/圆角/背景色。
是「破框九宫格」效果的底层画布——下游 breakout-compose 把抠好的人物叠上来。
"""
from flowx_client import emit_input_media, emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg
from image_batch import load_ids


def _parse_color(s, default="#FFFFFF"):
    """#RGB / #RRGGBB → (r,g,b)。"""
    s = (s or default).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bg 非法：{s!r}（用 #RRGGBB）")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _cover_square(img, side):
    """cover 语义缩放并居中裁成 side×side 正方形。"""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    k = max(side / w, side / h)
    nw, nh = max(1, round(w * k)), max(1, round(h * k))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - side) // 2, (nh - side) // 2
    return img.crop((left, top, left + side, top + side))


def _rounded(im, radius):
    """圆角化（radius<=0 原样返回）；返回 RGBA。"""
    im = im.convert("RGBA")
    if radius <= 0:
        return im
    from PIL import Image, ImageDraw
    mask = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, im.width - 1, im.height - 1], radius=int(radius), fill=255)
    im.putalpha(mask)
    return im


def compose(cells, cell=640, gap=10, margin=10, radius=0, bg="#FFFFFF",
            top_banner=0):
    """cells: 9 张 PIL 图（行优先）→ 九宫格 PIL 图（RGB）。

    画布 = margin*2 + cell*3 + gap*2 的正方形；top_banner>0 时顶部再加
    一块等宽背景色空区（人物破框超出的「画外空间」），画布变为矩形。
    """
    from PIL import Image
    if len(cells) != 9:
        raise ValueError(f"grid-compose 需要恰好 9 张图（3×3），收到 {len(cells)} 张")
    cell, gap, margin, top_banner = int(cell), int(gap), int(margin), int(top_banner)
    w = margin * 2 + cell * 3 + gap * 2
    h = w + max(0, top_banner)
    canvas = Image.new("RGBA", (w, h), _parse_color(bg) + (255,))
    for i, im in enumerate(cells):
        r, c = divmod(i, 3)
        x = margin + c * (cell + gap)
        y = top_banner + margin + r * (cell + gap)
        tile = _rounded(_cover_square(im, cell), radius)
        canvas.alpha_composite(tile, (x, y))
    return canvas.convert("RGB")


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    ids = load_ids(param("images"))
    # 输入原图物化：画布对比控件的「原图」侧（取第一张）
    emit_input_media(url, ids[0] if ids else "", tok)
    cells = [get_pimg(url, oid, tok, mode="RGB") for oid in ids]
    out = compose(cells, cell=param("cell_size", 640, cast=int),
                  gap=param("gap", 10, cast=int),
                  margin=param("margin", 10, cast=int),
                  radius=param("radius", 0, cast=int),
                  bg=param("bg", "#FFFFFF"),
                  top_banner=param("top_banner", 0, cast=int))
    oid = post_pimg(url, out, tok)
    print(f"[grid-compose] {len(cells)} 张 -> {out.width}x{out.height} image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height, count=len(cells))
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
