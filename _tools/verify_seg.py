"""B3 / G2 分割链落盘复核（dev-plan §49.6 A2/A3/A4/A5/A6 + A10）。

读 ~/flowx-e2e-out 里 4 条 e2e 的产物 + 从执行日志取算子回传的 JSON 负载，
做本机纯 PIL 量化复核；自动分割的 `selected` 用**取回单蒙版逐字节求并集**比对。

用法：python3 ~/workspace/github.com/LerkoX/.tmp/verify_seg.py
"""
import json
import pathlib
import re
import subprocess
import sys
import urllib.request

from PIL import Image, ImageChops

OUT = pathlib.Path.home() / "flowx-e2e-out"
BASE = "http://4dc21d2a.r7.cpolar.cn"
TOK = "30e494999da13134808ee280aea6dffd"
FAILED = 0
PASSED = 0


def check(name, ok, detail=""):
    global FAILED, PASSED
    if ok:
        PASSED += 1
        print(f"PASS {name}" + (f"  {detail}" if detail else ""))
    else:
        FAILED += 1
        print(f"FAIL {name}" + (f"  {detail}" if detail else ""))


def info(msg):
    print(f"    [信息] {msg}")


def latest(prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_+(\d{{8}}_\d{{6}})\.png$")
    fs = sorted([p for p in OUT.iterdir() if pat.match(p.name)], key=lambda p: p.stat().st_mtime)
    if not fs:
        sys.exit(f"找不到产物：{prefix}_*（先跑对应 e2e）")
    return fs[-1], Image.open(fs[-1])


def exec_outputs(eid):
    """从执行日志里取 flowx-yaml 块中的字段（值是按 YAML 规则引号包裹的标量）。"""
    p = subprocess.run(["flowx-studio", "execution", "logs", "--id", str(eid),
                        "--limit", "600"], capture_output=True, text=True)
    fields = {}
    pat = re.compile(r'\b([A-Za-z_][\w]*):\s("(?:[^"\\]|\\.)*")\s*$')
    for line in p.stdout.splitlines():
        m = pat.search(line)
        if m:
            try:
                fields[m.group(1)] = json.loads(m.group(2))
            except Exception:
                pass
    return fields


def fetch_image(oid):
    req = urllib.request.Request(f"{BASE}/images/{oid}",
                                 headers={"Authorization": "Bearer " + TOK})
    import io
    return Image.open(io.BytesIO(urllib.request.urlopen(req, timeout=120).read()))


def strict_binary(im):
    """蒙版必须是 0/255 二值（灰度 L）。"""
    vals = set(im.convert("L").get_flattened_data())
    return vals <= {0, 255}, sorted(vals)[:4]


def fg_ratio(im):
    return sum(1 for v in im.convert("L").get_flattened_data() if v > 127) / (im.size[0] * im.size[1])


def main():
    # ---------- A2/A3 点选与框选 ----------
    p_in, img_in = latest("seg_input")
    p_pt, img_pt = latest("seg_point_mask")
    p_annot, img_annot = latest("seg_point_annotated")
    p_bx, img_bx = latest("seg_box_mask")
    print(f"    输入 {p_in.name} {img_in.size} / 点选蒙版 {p_pt.name} / 框选蒙版 {p_bx.name}", flush=True)

    ok_pt, vals_pt = strict_binary(img_pt)
    ok_bx, vals_bx = strict_binary(img_bx)
    check("A2 点选蒙版为 0/255 二值灰度", ok_pt, f"取值 {vals_pt}")
    check("A3 框选蒙版为 0/255 二值灰度", ok_bx, f"取值 {vals_bx}")
    check("蒙版尺寸 == 输入尺寸", img_pt.size == img_in.size == img_bx.size,
          f"{img_pt.size} vs {img_in.size} vs {img_bx.size}")
    f_pt, f_bx = fg_ratio(img_pt), fg_ratio(img_bx)
    check("前景占比落在合理区间（0.02~0.95）", 0.02 < f_pt < 0.95 and 0.02 < f_bx < 0.95,
          f"点选={f_pt:.3f} 框选={f_bx:.3f}")
    diff = ImageChops.difference(img_pt.convert("L"), img_bx.convert("L"))
    n_diff = sum(1 for v in diff.get_flattened_data() if v > 0)
    check("不同提示（点选 vs 框选）得到不同蒙版", n_diff > 0,
          f"差异像素={n_diff}（{n_diff / (img_pt.size[0] * img_pt.size[1]) * 100:.1f}%）")
    check("标注图尺寸 == 输入尺寸", img_annot.size == img_in.size, str(img_annot.size))

    # A3：框选蒙版应基本落在提示框内（框=(80,60,430,330)，容忍 4px 外扩）
    x1, y1, x2, y2 = 80, 60, 430, 330
    W, H = img_bx.size
    outside = inside = 0
    for i, v in enumerate(img_bx.convert("L").get_flattened_data()):
        if v > 127:
            x, y = i % W, i // W
            if x1 - 4 <= x <= x2 + 4 and y1 - 4 <= y <= y2 + 4:
                inside += 1
            else:
                outside += 1
    ratio = outside / max(1, inside + outside)
    check("A3 框选蒙版 ≥95% 落在提示框内（4px 容忍）", ratio <= 0.05,
          f"框外占比={ratio * 100:.2f}%（in={inside} out={outside}）")

    # ---------- A6 自动分割：keep 子集并集逐字节一致 ----------
    p_mont, img_mont = latest("seg_auto_montage")
    p_sel, img_sel = latest("seg_auto_selected")
    p_ov, img_ov = latest("seg_auto_overlay")
    f = exec_outputs(654)
    ids = json.loads(f.get("mask_ids", "[]").lstrip("@")) if f.get("mask_ids") else []
    sizes = json.loads(f.get("sizes", "[]").lstrip("@")) if f.get("sizes") else []
    # keep 是**参数**（不在 emit 输出里）⇒ 从流水线 YAML 取（与 e2e 同源）
    _y = pathlib.Path.home() / "workspace/github.com/LerkoX/flowx-pixelforge/pipeline/seg-auto-e2e-local.yaml"
    m_keep = re.search(r'^\s*keep:\s*"?([0-9,\s]*)"?\s*$', _y.read_text(), re.M)
    keep = [int(x) for x in (m_keep.group(1) if m_keep else "").split(",") if x.strip().isdigit()]
    count = int(f.get("count", "0") or 0)
    cols = int(f.get("montage_cols", "1") or 1)
    info(f"exec 654：count={count} montage_cols={cols} keep={keep} 单蒙版 id 数={len(ids)}")
    check("A6 自动分割回传的单蒙版 id 数 == count", len(ids) == count,
          f"{len(ids)} vs {count}")
    check("A6 蒙版网格图布局与 count/cols 自洽",
          img_mont.size == (160 * cols, 160 * ((count + cols - 1) // cols)),
          f"{img_mont.size} vs 期望 {(160 * cols, 160 * ((count + cols - 1) // cols))}")
    if keep and ids:
        union = None
        for i in keep:
            if i >= len(ids):
                continue
            m = fetch_image(ids[i]).convert("L")
            union = m if union is None else ImageChops.lighter(union, m)
        if union is not None:
            same = union.size == img_sel.size and union.tobytes() == img_sel.convert("L").tobytes()
            nz = sum(1 for v in union.get_flattened_data() if v > 0)
            check("A6 selected == 取回单蒙版按 keep 求并集（逐字节）", same,
                  f"并集非零像素={nz}，selected 非零="
                  f"{sum(1 for v in img_sel.convert('L').get_flattened_data() if v > 0)}")
    sel_fg = fg_ratio(img_sel)
    info(f"selected 前景占比={sel_fg:.3f}；union_area 回传={f.get('union_area')}；"
         f"bbox={f.get('bbox')}")
    check("A6 selected 前景占比与 union_area 自洽",
          abs(sel_fg * img_sel.size[0] * img_sel.size[1] - float(f.get("union_area", 0))) <= 1,
          f"{sel_fg * img_sel.size[0] * img_sel.size[1]:.0f} vs {f.get('union_area')}")
    check("叠加预览图尺寸 == 输入尺寸", img_ov.size == img_in.size, str(img_ov.size))

    # ---------- A5 文本 → 像素级蒙版 ----------
    p_tmask, img_tmask = latest("seg_text_mask")
    p_tannot, img_tannot = latest("seg_text_annotated")
    g = exec_outputs(655)
    boxes = json.loads(g.get("boxes", "[]").lstrip("@")) if g.get("boxes") else []
    tcount = int(g.get("count", "0") or 0)
    info(f"exec 655：count={tcount} boxes={len(boxes)} sam_model={g.get('sam_model')!r}")
    check("A5 文本检测命中数 == 框数 == count", len(boxes) == tcount and tcount >= 1,
          f"boxes={len(boxes)} count={tcount}")
    # 框矩形并集 vs SAM 像素级蒙版：后者应更小（边界贴合）
    W2, H2 = img_tmask.size
    box_union = bytearray(W2 * H2)
    for b in boxes:
        bx1, by1, bx2, by2 = (int(round(v)) for v in b)
        for y in range(max(0, by1), min(H2, by2)):
            base = y * W2
            for x in range(max(0, bx1), min(W2, bx2)):
                box_union[base + x] = 1
    nb = sum(box_union)
    ns = sum(1 for v in img_tmask.convert("L").get_flattened_data() if v > 127)
    check("A5 SAM 像素级蒙版严格小于框矩形并集（边界贴合）", 0 < ns < nb,
          f"SAM={ns}px 框并集={nb}px（省 {(1 - ns / max(1, nb)) * 100:.1f}%）")
    # 精修蒙版应基本落在框内（容忍 8px）
    out8 = in8 = 0
    px = img_tmask.convert("L").get_flattened_data()
    for i, v in enumerate(px):
        if v > 127:
            x, y = i % W2, i // W2
            hit = False
            for b in boxes:
                if b[0] - 8 <= x <= b[2] + 8 and b[1] - 8 <= y <= b[3] + 8:
                    hit = True
                    break
            in8 += 1 if hit else 0
            out8 += 0 if hit else 1
    check("A5 精修蒙版 ≥97% 落在检测框内（8px 容忍）",
          out8 / max(1, in8 + out8) <= 0.03,
          f"框外占比={out8 / max(1, in8 + out8) * 100:.2f}%")
    check("标注图尺寸 == 输入尺寸", img_tannot.size == img_in.size, str(img_tannot.size))
    info(f"产物：{p_mont.name} / {p_sel.name} / {p_ov.name} / {p_tmask.name} / {p_tannot.name}")

    print(f"\nFAILED = {FAILED}   PASSED = {PASSED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
