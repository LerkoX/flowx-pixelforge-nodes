"""拼图（联系表）共享件（B5/52.2）：N 张图 → 一张网格图。纯 PIL，零 numpy。

被 `image-montage` 节点使用。**尺寸公式是公开契约**（复核脚本按它对表）：
    W = cols*cell_w + (cols+1)*pad
    H = rows*cell_h + (rows+1)*pad + rows*label_h
（`label_h` 仅在 label != none 时非零）

为什么需要：Studio 没有循环概念，把 N 张图合成一张再进单图链（或直接人眼验收）是唯一正路。
"""
import io
import math

from PIL import Image, ImageDraw

from image_batch import get_pimg_bytes


def _color(s, default=(22, 22, 24)):
    t = str(s or "").strip().lstrip("#")
    if len(t) == 3:
        t = "".join(c * 2 for c in t)
    if len(t) != 6:
        return default
    try:
        return tuple(int(t[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def fetch(url, oid, tok=None):
    """对象 ID → PIL.Image（走 HTTP，PNG/JPEG 皆可）。"""
    data = get_pimg_bytes(url, oid, tok)
    return Image.open(io.BytesIO(data))


def _fit(img, w, h, mode, bg=(22, 22, 24)):
    """fit=等比装入（留边填 bg）；cover=等比铺满后居中裁切。

    ⚠️ 留边必须**填背景色**：早期版本用 `crop` 补齐（小图裁大框）会让 PIL 用**黑色**
    填边（本机单测抓到）——`fit` 的正确做法是贴到 bg 画布上居中。
    """
    iw, ih = img.size
    cell = Image.new("RGB", (w, h), bg)
    if iw <= 0 or ih <= 0:
        return cell
    k = (min if mode != "cover" else max)(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * k))), max(1, int(round(ih * k)))
    r = img.convert("RGB").resize((nw, nh), Image.LANCZOS)
    if mode == "cover":
        left, top = (nw - w) // 2, (nh - h) // 2
        r = r.crop((left, top, left + w, top + h))
        cell.paste(r, (0, 0))
    else:
        cell.paste(r, ((w - nw) // 2, (h - nh) // 2))
    return cell


def auto_cols(n):
    """自动列数 ≈ sqrt(n)，至少 1 列。"""
    return max(1, int(math.ceil(math.sqrt(max(1, int(n))))))


def montage(url, ids, tok=None, cols=0, cell_w=0, cell_h=0, pad=10, label="none",
            label_h=18, bg="#161618", fit="fit", names=None):
    """→ (PIL 拼图, grid JSON `[cols, rows]`)。

    cols=0 自动；cell_w/cell_h=0 时取所有图的**最大**宽/高（不缩放任何图，A3 判据成立）。
    label ∈ none|index|name；`names` 与 ids 同序（通常来自 image-batch-load 的 names 输出）
    —— **label=name 一定要传它**，否则只能打对象 ID（对人没用）。
    """
    if not ids:
        raise ValueError("image-montage: images 为空（先给 image-batch-load 的输出）")
    mode = str(fit or "fit").strip().lower()
    if mode not in ("fit", "cover"):
        raise ValueError(f"image-montage: fit 非法 {fit!r}（fit/cover）")
    lab = str(label or "none").strip().lower()
    if lab not in ("none", "index", "name"):
        raise ValueError(f"image-montage: label 非法 {label!r}（none/index/name）")
    imgs = [fetch(url, i, tok) for i in ids]
    nm = [str(x) for x in (names or [])]
    if lab == "name" and len(nm) != len(ids):
        raise ValueError(f"image-montage: label=name 需要 names（{len(ids)} 个），"
                         f"实际 {len(nm)} 个（接 image-batch-load 的 names 输出）")
    n = len(imgs)
    c = int(cols or 0) or auto_cols(n)
    rows = int(math.ceil(n / c))
    cw = int(cell_w or 0) or max(i.width for i in imgs)
    ch = int(cell_h or 0) or max(i.height for i in imgs)
    p = max(0, int(pad or 0))
    lh = max(0, int(label_h or 0)) if lab != "none" else 0
    W = c * cw + (c + 1) * p
    H = rows * ch + (rows + 1) * p + rows * lh
    canvas = Image.new("RGB", (W, H), _color(bg))
    d = ImageDraw.Draw(canvas)
    for i, im in enumerate(imgs):
        r, col = divmod(i, c)
        x = p + col * (cw + p)
        y = p + r * (ch + p + lh)
        canvas.paste(_fit(im, cw, ch, mode, _color(bg)), (x, y))
        if lh:
            txt = str(i) if lab == "index" else nm[i]
            d.text((x + 2, y + ch + 2), txt[:40], fill=(235, 235, 235))
    import json
    return canvas, "@" + json.dumps([c, rows])


def upload_pil(url, pil, tok=None):
    """PIL → 对象仓库（PNG 编码），返回对象 ID。"""
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    from flowx_client import post_bytes
    r = post_bytes(url, "/images", buf.getvalue(), tok, timeout=300,
                   content_type="image/png")
    oid = r.get("id") or (r.get("outputs") or {}).get("id")
    if not oid:
        raise RuntimeError(f"image-montage: 上传拼图失败：{r}")
    return oid
