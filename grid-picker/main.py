"""grid-picker：九宫格选图编排（上传 9 张 → 弹窗编排格位/裁剪中心 → 合成底图）。

画布弹窗写回两个参数：files（9 个本机路径 JSON，格位顺序 = 行优先）与
crops（每格 cover 裁切焦点 {fx,fy}，0~1，默认 0.5=居中）。执行时按焦点
裁方 → 可选 beautify.auto 美化（服务端算子）→ 拼 3×3 底图（可调格尺寸/
间隙/边距/圆角/背景色/顶部白区）→ 喂 breakout-compose。
"""
import json
import os

from flowx_client import (call_op, emit, emit_result_preview, ensure_plugin,
                          param, ref, token)
from image_batch import dumps, get_pimg_bytes, loads
from image_montage import upload_pil

OP = "beautify.auto"


def parse_files(raw):
    """files 参数 → 本机绝对路径列表（保留顺序，过滤不存在）。空/非法给 []。"""
    out = []
    for p in loads(raw):
        full = os.path.abspath(os.path.expanduser(str(p)))
        if os.path.isfile(full):
            out.append(full)
    return out


def parse_crops(raw, n):
    """crops 参数 → n 个 (fx,fy)；缺省/越界回退 0.5。"""
    default = [(0.5, 0.5)] * n
    try:
        arr = json.loads(raw) if raw else []
    except Exception:
        return default
    out = []
    for i in range(n):
        try:
            fx = min(1.0, max(0.0, float(arr[i]["fx"])))
            fy = min(1.0, max(0.0, float(arr[i]["fy"])))
            out.append((fx, fy))
        except Exception:
            out.append((0.5, 0.5))
    return out


def cover_crop_focus(img, side, fx=0.5, fy=0.5):
    """cover 裁正方形：短边充满，长边按焦点 (fx,fy) 定位裁剪窗（0=顶/左，1=底/右）。"""
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    m = min(w, h)
    if w > h:
        x0 = int(round(fx * (w - m)))
        img = img.crop((x0, 0, x0 + m, m))
    elif h > w:
        y0 = int(round(fy * (h - m)))
        img = img.crop((0, y0, m, y0 + m))
    return img.resize((side, side), Image.LANCZOS)


def _parse_color(s, default="#FFFFFF"):
    s = (s or default).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bg 非法：{s!r}（用 #RRGGBB）")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _rounded(im, radius):
    im = im.convert("RGBA")
    if radius <= 0:
        return im
    from PIL import Image, ImageDraw
    mask = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, im.width - 1, im.height - 1], radius=int(radius), fill=255)
    im.putalpha(mask)
    return im


def compose(cells, cell=640, gap=10, margin=10, radius=0, bg="#FFFFFF", top_banner=0):
    """9 格 PIL 图（行优先）→ 底图（与 grid-compose 同版式）。"""
    from PIL import Image
    if len(cells) != 9:
        raise ValueError(f"grid-picker 需要恰好 9 张图（3×3），收到 {len(cells)} 张")
    cell, gap, margin, top_banner = int(cell), int(gap), int(margin), int(top_banner)
    w = margin * 2 + cell * 3 + gap * 2
    h = w + max(0, top_banner)
    canvas = Image.new("RGBA", (w, h), _parse_color(bg) + (255,))
    for i, im in enumerate(cells):
        r, c = divmod(i, 3)
        canvas.alpha_composite(_rounded(im, radius),
                               (margin + c * (cell + gap),
                                top_banner + margin + r * (cell + gap)))
    return canvas.convert("RGB")


def main():
    from PIL import Image
    url = param("service_url").rstrip("/")
    tok = token()
    files = parse_files(param("files", ""))
    if len(files) != 9:
        raise RuntimeError(f"grid-picker 需要恰好 9 张底图（当前 {len(files)} 张）"
                           "——点节点上「✏️ 编排九宫格」上传并编排后再运行")
    crops = parse_crops(param("crops", ""), 9)
    cell = param("cell_size", 640, cast=int)
    beautify = str(param("beautify", "false")).lower() in ("1", "true", "yes", "on")
    op = None
    bkw = {}
    if beautify:
        plugin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_op.py")
        op = ensure_plugin(url, OP, plugin_path=plugin, tok=tok)
        bkw = dict(crop=0,  # 裁方由本节点按焦点做，美化只出图
                   whiten=param("whiten", 0.5, cast=float),
                   smooth=param("smooth", 0.4, cast=float),
                   eye=param("eye", 0.12, cast=float),
                   slim=param("slim", 0.10, cast=float),
                   det_thresh=param("det_thresh", 0.3, cast=float))
    cells = []
    ids = []  # 每格裁方（+美化）后的对象 ID，作为列表输出供下游精修/风格化
    for i, path in enumerate(files):
        fx, fy = crops[i]
        im = cover_crop_focus(Image.open(path), cell, fx, fy)
        if op:
            oid = upload_pil(url, im, tok)
            r = call_op(url, op, {"image": ref(oid), **bkw}, tok)
            import io
            im = Image.open(io.BytesIO(get_pimg_bytes(url, r["image"], tok))).convert("RGB")
            ids.append(r["image"])
            print(f"[grid-picker] {i+1}/9 裁方+美化 {os.path.basename(path)}", flush=True)
        else:
            ids.append(upload_pil(url, im, tok))
            print(f"[grid-picker] {i+1}/9 裁方 {os.path.basename(path)} "
                  f"focus=({fx:.2f},{fy:.2f})", flush=True)
        cells.append(im)
    images_json = dumps(ids)
    if str(param("compose", "true")).strip().lower() not in ("1", "true", "yes", "on"):
        # 不合成：只出 9 张图列表，由下游精修/风格化后再交 grid-compose 拼图
        print(f"[grid-picker] 9 张 -> images 列表（compose=off，不合成底图）", flush=True)
        emit(images=images_json, count=9, image="", width=0, height=0)
        return
    out = compose(cells, cell=cell,
                  gap=param("gap", 10, cast=int),
                  margin=param("margin", 10, cast=int),
                  radius=param("radius", 0, cast=int),
                  bg=param("bg", "#FFFFFF"),
                  top_banner=param("top_banner", 0, cast=int))
    oid = upload_pil(url, out, tok)
    print(f"[grid-picker] 9 张 -> {out.width}x{out.height} image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height, count=9, images=images_json)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
