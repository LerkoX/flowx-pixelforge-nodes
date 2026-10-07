"""mask-paint：画布手绘蒙版（对标 ComfyUI MaskEditor）。

笔画以矢量存于 strokes_json 参数（widget 蒙版编辑器写入）：
  {"strokes":[{"type":"brush","size":40,"points":[[x,y],...]},
              {"type":"rect","x":..,"y":..,"w":..,"h":..},
              {"type":"erase","size":40,"points":[...]}],
   "image_size":[w,h]}        # 绘制时的参照分辨率

执行时按原图实际分辨率缩放光栅化为灰度蒙版（255=重绘/0=保留），
mask 约定与 mask-from-image 一致 = 内容为灰度的 IMAGE 对象。
"""
import json

from flowx_client import emit, param, token
from image_io import get_pimg, post_pimg, thumb_data_url

DEFAULT_REF = (512, 512)


def parse_strokes(raw):
    """解析 strokes_json；空串/坏 JSON → 空笔画。返回 (strokes, ref_w, ref_h)。"""
    if not raw:
        return [], *DEFAULT_REF
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return [], *DEFAULT_REF
    strokes = data.get("strokes") or []
    size = data.get("image_size") or list(DEFAULT_REF)
    try:
        rw, rh = max(1, int(size[0])), max(1, int(size[1]))
    except (TypeError, ValueError, IndexError):
        rw, rh = DEFAULT_REF
    return [s for s in strokes if isinstance(s, dict)], rw, rh


def rasterize(strokes, width, height, ref_w=None, ref_h=None):
    """矢量笔画 → 灰度蒙版（PIL L 模式，255=重绘/0=保留）。

    笔画坐标系 = (ref_w, ref_h) 参照分辨率，自动缩放到 (width, height)。
    brush/erase 为折线圆头笔刷（size=直径），rect 为实心矩形；erase 后画压先画。
    """
    from PIL import Image, ImageDraw

    ref_w = ref_w or width
    ref_h = ref_h or height
    sx = width / ref_w
    sy = height / ref_h
    sm = (sx + sy) / 2  # 笔刷直径按两轴均值缩放
    img = Image.new("L", (width, height), 0)
    dr = ImageDraw.Draw(img)
    for st in strokes:
        kind = st.get("type")
        fill = 0 if kind == "erase" else 255
        if kind in ("brush", "erase"):
            size = max(1, round(float(st.get("size", 40)) * sm))
            r = size / 2
            pts = [(float(p[0]) * sx, float(p[1]) * sy)
                   for p in (st.get("points") or []) if len(p) >= 2]
            if not pts:
                continue
            if len(pts) == 1:
                x, y = pts[0]
                dr.ellipse([x - r, y - r, x + r, y + r], fill=fill)
            else:
                # joint="curve" 平滑拐角；两端补圆头（PIL line 默认平头帽）
                dr.line(pts, fill=fill, width=size, joint="curve")
                for (x, y) in (pts[0], pts[-1]):
                    dr.ellipse([x - r, y - r, x + r, y + r], fill=fill)
        elif kind == "rect":
            x = float(st.get("x", 0)) * sx
            y = float(st.get("y", 0)) * sy
            w = float(st.get("w", 0)) * sx
            h = float(st.get("h", 0)) * sy
            dr.rectangle([min(x, x + w), min(y, y + h),
                          max(x, x + w), max(y, y + h)], fill=fill)
    return img


def coverage(mask):
    """白区（重绘区）像素占比 0~1。"""
    hist = mask.histogram()
    total = mask.width * mask.height
    return sum(hist[128:]) / total if total else 0.0


def overlay_thumb(im, mask):
    """红罩叠加原图的预览缩略图（执行结果直接可核对涂抹位置）。"""
    from PIL import Image

    base = im.convert("RGB")
    red = Image.new("RGB", base.size, (244, 63, 94))
    a = mask.point(lambda v: int(v * 0.6))
    return Image.composite(red, base, a)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    strokes, rw, rh = parse_strokes(param("strokes_json", ""))

    im = get_pimg(url, param("image"), tok)
    mask = rasterize(strokes, im.width, im.height, rw, rh)
    cov = coverage(mask)
    oid = post_pimg(url, mask, tok)
    print(f"[mask-paint] image={param('image')} strokes={len(strokes)} "
          f"ref={rw}x{rh} -> mask={oid} {im.width}x{im.height} "
          f"coverage={cov:.1%}", flush=True)
    emit(mask=oid, width=im.width, height=im.height, coverage=round(cov, 4),
         thumb_b64=thumb_data_url(overlay_thumb(im, mask)))


if __name__ == "__main__":
    main()
