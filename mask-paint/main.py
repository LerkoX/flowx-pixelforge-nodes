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
import os

from flowx_client import emit, emit_result_preview, param, post_bytes, token
from image_io import get_pimg, post_pimg

DEFAULT_REF = (512, 512)
_MAX_BYTES = 32 * 1024 * 1024  # 与服务端 MAX_UPLOAD_MB 默认值对齐


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


def upload_local(url, path, tok):
    """本地图片 → 推理服务图像对象（等价 load-image 的上传链路）。

    返回 (image_id, width, height)。"""
    if not os.path.isfile(path):
        raise RuntimeError(f"image file not found: {path}")
    with open(path, "rb") as f:
        data = f.read()
    if len(data) > _MAX_BYTES:
        raise RuntimeError(f"image too large: {len(data)} bytes > {_MAX_BYTES}")
    ctype = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    resp = post_bytes(url, "/images", data, tok, timeout=300, content_type=ctype)
    return resp["id"], resp.get("width"), resp.get("height")


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    strokes, rw, rh = parse_strokes(param("strokes_json", ""))

    # 底图取法：image（上游对象 ID）优先；缺省时读 image_path 本地图自行上传
    image_id = param("image", "").strip()
    if image_id:
        im = get_pimg(url, image_id, tok)
        width, height = im.size
    else:
        path = os.path.expanduser(param("image_path", "").strip())
        if not path:
            raise RuntimeError("mask-paint needs image (服务端对象 ID) or "
                               "image_path (本地图片路径)")
        image_id, width, height = upload_local(url, path, tok)
        if not width or not height:
            width, height = get_pimg(url, image_id, tok).size
        print(f"[mask-paint] uploaded {path} -> image={image_id} {width}x{height}",
              flush=True)

    mask = rasterize(strokes, width, height, rw, rh)
    cov = coverage(mask)
    oid = post_pimg(url, mask, tok)
    print(f"[mask-paint] image={image_id} strokes={len(strokes)} "
          f"ref={rw}x{rh} -> mask={oid} {width}x{height} "
          f"coverage={cov:.1%}", flush=True)
    emit(image=image_id, mask=oid, width=width, height=height,
         coverage=round(cov, 4))
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
