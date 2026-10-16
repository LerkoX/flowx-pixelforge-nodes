"""image-watermark：文字/Logo 水印（电商白底图、详情页、样张交付的最后一公里）。

纯节点侧 PIL（**零 numpy** ⇒ 手机本地执行器也能跑）。要点：
- `position` 九宫格定位 + `margin` 边距；`tile=True` 时按 `angle` 平铺（防盗图样张）。
- `opacity` 0~1 控制水印整体不透明度（用 RGBA 合成，不改动非水印区域像素）。
- `font_size=0` 自动取 `max(12, 短边/18)`；字体默认取**节点包内自带**的
  `fonts/DroidSansChinese.ttf`（含 CJK），`font_path` 可显式覆盖 ⇒ local 与 docker
  渲染逐字节一致（此前两边字体不同，产物不可比，见 dev-plan §50.9 与 dev-log §60）。
- 文字带 `stroke_width`/`stroke_color` 描边，深浅背景都可读。
- 输出 `box`（水印覆盖区域 x,y,w,h）⇒ 复核判据"非水印区逐字节不变"直接可用。

⚠️ 与 ComfyUI 的 ImageCompositeMasked/水印节点等价能力：定位、透明度、平铺、旋转、Logo 缩放。
"""
import os

from PIL import Image, ImageDraw, ImageFont

from flowx_client import emit_input_media, emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg, pbool

POSITIONS = ("top-left", "top-center", "top-right",
             "middle-left", "center", "middle-right",
             "bottom-left", "bottom-center", "bottom-right")

# 随节点包分发的固定字体（dev-log §60：local/docker 字体不同会让文本渲染产物不可比，
# A7 逐字节判据对文本节点不成立）。**首选**它 ⇒ 同一节点包在手机与容器里渲染一致；
# `font_path` 参数仍可覆盖（用户自带字体时按显式路径走）。
_BUNDLED_FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "fonts", "DroidSansChinese.ttf")

# 常见字体候选（按顺序取第一个存在的；都没有则用 Pillow 内置可缩放默认字体）
FONT_CANDIDATES = (
    _BUNDLED_FONT,
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/system/fonts/DroidSansChinese.ttf",
    "/system/fonts/NotoSansCJK-Regular.ttc",
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/arial.ttf",
)


def _parse_color(s, default=(255, 255, 255)):
    """'#RRGGBB' / '#RRGGBBAA' / 'r,g,b' → RGBA 元组。"""
    t = str(s or "").strip()
    if not t:
        return default if len(default) == 4 else (*default, 255)
    if t.startswith("#"):
        h = t[1:]
        if len(h) == 6:
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (255,)
        if len(h) == 8:
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4, 6))
        raise ValueError(f"image-watermark: 颜色非法 {s!r}（用 #RRGGBB 或 #RRGGBBAA）")
    parts = [p for p in t.replace(" ", ",").split(",") if p != ""]
    if len(parts) not in (3, 4):
        raise ValueError(f"image-watermark: 颜色非法 {s!r}（用 r,g,b 或 r,g,b,a）")
    vals = [int(float(p)) for p in parts]
    if any(v < 0 or v > 255 for v in vals):
        raise ValueError(f"image-watermark: 颜色分量需 0~255，got {s!r}")
    return tuple(vals) + ((255,) if len(vals) == 3 else ())


def _load_font(font_path, size):
    """字体：显式路径 > 候选列表 > Pillow 内置（可缩放，ASCII）。"""
    cands = ([font_path] if font_path else []) + list(FONT_CANDIDATES)
    for p in cands:
        try:
            if p and os.path.isfile(p):
                return ImageFont.truetype(p, int(size)), p
        except Exception:
            continue
    try:
        return ImageFont.load_default(size=int(size)), "(default)"
    except TypeError:                      # 老 Pillow 无 size 参数
        return ImageFont.load_default(), "(default-small)"


def _text_layer(text, font, color, stroke_width, stroke_fill):
    """把文字画到紧贴的 RGBA 图层，返回 (layer, (w,h))。"""
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    box = probe.textbbox((0, 0), text, font=font, stroke_width=int(stroke_width))
    pad = int(stroke_width) + 2
    w, h = box[2] - box[0] + pad * 2, box[3] - box[1] + pad * 2
    layer = Image.new("RGBA", (max(1, w), max(1, h)), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((pad - box[0], pad - box[1]), text, font=font,
                               fill=color, stroke_width=int(stroke_width),
                               stroke_fill=stroke_fill)
    return layer, (w, h)


def _logo_layer(logo, scale, opacity):
    """Logo 图层：按 scale 缩放（0=自动取画布短边的 12%）。"""
    im = logo.convert("RGBA")
    if scale <= 0:
        return im, im.size
    w, h = im.size
    nw, nh = max(1, int(w * float(scale))), max(1, int(h * float(scale)))
    return im.resize((nw, nh), Image.LANCZOS), (nw, nh)


def _pos_xy(position, canvas, item, margin):
    """九宫格 → 左上角坐标。"""
    W, H = canvas
    w, h = item
    m = int(margin)
    x = {"left": m, "center": (W - w) // 2, "right": W - w - m}[
        position.split("-")[-1] if position != "center" else "center"]
    y = {"top": m, "middle": (H - h) // 2, "bottom": H - h - m}[
        position.split("-")[0] if position != "center" else "middle"]
    return int(x), int(y)


def _make_item(text, logo, font_path, font_size, color, stroke_width, stroke_color,
               logo_scale, canvas):
    """生成水印图元（RGBA）与其尺寸。"""
    if logo is not None:
        return _logo_layer(logo, logo_scale, None)
    if not str(text or ""):
        raise ValueError("image-watermark: text 与 logo 至少要给一个")
    W, H = canvas
    size = int(font_size) if int(font_size) > 0 else max(12, min(W, H) // 18)
    font, used = _load_font(font_path, size)
    layer, dim = _text_layer(str(text), font, color, stroke_width, stroke_color)
    print(f"[image-watermark] font={used} size={size}", flush=True)
    return layer, dim


def process(image, text="", logo=None, position="bottom-right", margin=24,
            opacity=0.6, font_size=0, font_path="", color="#FFFFFF",
            stroke_width=2, stroke_color="#000000", logo_scale=0.0,
            tile=False, angle=0, tile_gap=0):
    """合成水印，返回 (结果图, box(x,y,w,h), 水印图元尺寸)。纯 PIL，无 numpy。"""
    pos = str(position or "bottom-right").strip().lower()
    if pos not in POSITIONS:
        raise ValueError(f"image-watermark: position 非法 {position!r}，可选 "
                         f"{list(POSITIONS)}")
    op = float(opacity)
    if not (0.0 < op <= 1.0):
        raise ValueError(f"image-watermark: opacity 非法 {opacity!r}（需 0 < o <= 1）")
    col = _parse_color(color, (255, 255, 255, 255))
    scol = _parse_color(stroke_color, (0, 0, 0, 255))

    base = image.convert("RGB")
    W, H = base.size
    item, (iw, ih) = _make_item(text, logo, font_path, font_size, col,
                                stroke_width, scol, logo_scale, (W, H))
    ang = int(angle) % 360
    if ang:
        item = item.rotate(ang, expand=True, resample=Image.BICUBIC)
        iw, ih = item.size
    if op < 1.0:                                   # 整体不透明度：只缩 alpha
        alpha = item.getchannel("A").point(lambda v: int(v * op))
        item.putalpha(alpha)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if pbool(tile):
        gap = int(tile_gap)
        step_x, step_y = iw + gap, ih + gap
        for y in range(-ih, H + ih, max(1, step_y)):
            for x in range(-iw, W + iw, max(1, step_x)):
                overlay.alpha_composite(item, (x, y))
        box = (0, 0, W, H)
    else:
        x, y = _pos_xy(pos, (W, H), (iw, ih), margin)
        overlay.alpha_composite(item, (max(0, x), max(0, y)))
        box = (max(0, x), max(0, y), min(iw, W - max(0, x)), min(ih, H - max(0, y)))

    out = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
    return out, box, (iw, ih)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    # 输入原图物化：画布对比控件的「原图」侧（推理服务对象 GC/TTL 后
    # 必然 404，必须在执行当时落一份到 Studio 媒体缓存）
    emit_input_media(url, param("image"), tok)
    # 先取参数（本地桩冒烟测试要求 main() 里所有名字都有定义，见 test_main.py 第 6 节）
    image_id = param("image")
    logo_id = param("logo", "")
    position = param("position", "bottom-right")
    margin = param("margin", 24, cast=int)
    opacity = param("opacity", 0.6, cast=float)
    font_size = param("font_size", 0, cast=int)
    font_path = param("font_path", "")
    color = param("color", "#FFFFFF")
    stroke_width = param("stroke_width", 2, cast=int)
    stroke_color = param("stroke_color", "#000000")
    logo_scale = param("logo_scale", 0.0, cast=float)
    tile = pbool(param("tile", "false"))
    angle = param("angle", 0, cast=int)
    tile_gap = param("tile_gap", 0, cast=int)
    im = get_pimg(url, image_id, tok, mode="RGB")
    logo = get_pimg(url, logo_id, tok, mode="RGBA") if logo_id else None
    out, box, item = process(
        im, text=param("text", ""), logo=logo, position=position, margin=margin,
        opacity=opacity, font_size=font_size, font_path=font_path, color=color,
        stroke_width=stroke_width, stroke_color=stroke_color,
        logo_scale=logo_scale, tile=tile, angle=angle, tile_gap=tile_gap,
    )
    oid = post_pimg(url, out, tok)
    print(f"[image-watermark] {im.size} text={param('text', '')!r} "
          f"logo={'有' if logo_id else '无'} pos={position} opacity={opacity} "
          f"tile={tile} box={box} -> {oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height,
         box=",".join(str(v) for v in box),
         box_w=box[2], box_h=box[3], item_w=item[0], item_h=item[1])
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
