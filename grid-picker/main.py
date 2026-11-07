"""grid-picker：九宫格选图编排（上传 9 张 → 弹窗编排格位/裁剪中心 → 逐格裁方出图）。

画布弹窗写回两个参数：files（9 个本机路径 JSON，格位顺序 = 行优先）与
crops（每格 cover 裁切焦点 {fx,fy}，0~1，默认 0.5=居中）。执行时按焦点
裁方 → 可选 beautify.auto 美化（服务端算子）→ 以 img1~img9 独立输出
（+ images 列表）供下游逐格精修/风格化；拼接由 grid-compose 完成。
"""
import json
import os

from flowx_client import (call_op, emit, ensure_plugin, param, ref, token)
from image_batch import dumps, loads
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
    ids = []  # 每格裁方（+美化）后的对象 ID
    for i, path in enumerate(files):
        fx, fy = crops[i]
        im = cover_crop_focus(Image.open(path), cell, fx, fy)
        if op:
            oid = upload_pil(url, im, tok)
            r = call_op(url, op, {"image": ref(oid), **bkw}, tok)
            ids.append(r["image"])
            print(f"[grid-picker] {i+1}/9 裁方+美化 {os.path.basename(path)}", flush=True)
        else:
            ids.append(upload_pil(url, im, tok))
            print(f"[grid-picker] {i+1}/9 裁方 {os.path.basename(path)} "
                  f"focus=({fx:.2f},{fy:.2f})", flush=True)
    kw = {"images": dumps(ids), "count": 9}
    kw.update({f"img{i+1}": oid for i, oid in enumerate(ids)})
    print("[grid-picker] 9 张 -> img1~img9 + images 列表（拼接交 grid-compose）", flush=True)
    emit(**kw)


if __name__ == "__main__":
    main()
