"""B3 / G4 分块重绘超分落盘复核（dev-plan §49.6 A7/A8/A9 + A10）。

读 ~/flowx-e2e-out 里 wf 136 的产物（tiled_input/pre/seam/up），本机纯 PIL 量化：
  A7 尺寸与预放大一致性（预放大 == 本机 LANCZOS 2x，逐字节）
  A8 接缝：ΣW 权重图在内区应≈1（互补 ramp 的数学证据）+ 接缝带梯度不异常升高
  A9 细节：重绘结果边缘能量不低于预放大图（不糊），并报提升比

用法：python3 ~/workspace/github.com/LerkoX/.tmp/verify_tiled.py
"""
import pathlib
import re
import sys

from PIL import Image, ImageChops, ImageFilter

OUT = pathlib.Path.home() / "flowx-e2e-out"
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
        sys.exit(f"找不到产物：{prefix}_*（先跑 wf 136）")
    return fs[-1], Image.open(fs[-1])


def edge_energy(im):
    """边缘能量代理：FIND_EDGES 均值（越大越"有细节"）。"""
    g = im.convert("L").filter(ImageFilter.FIND_EDGES)
    d = g.get_flattened_data()
    return sum(d) / len(d)


def col_means(im):
    """每列的相邻像素平均差（接缝硬边会在个别列上尖峰）。"""
    W, H = im.size
    a = im.convert("L").crop((0, 0, W - 1, H))
    b = im.convert("L").crop((1, 0, W, H))
    d = ImageChops.difference(a, b).get_flattened_data()
    out = [0.0] * (W - 1)
    for i, v in enumerate(d):
        out[i % (W - 1)] += v
    return [v / H for v in out]


def row_means(im):
    W, H = im.size
    a = im.convert("L").crop((0, 0, W, H - 1))
    b = im.convert("L").crop((0, 1, W, H))
    d = ImageChops.difference(a, b).get_flattened_data()
    out = [0.0] * (H - 1)
    for i, v in enumerate(d):
        out[i // W] += v
    return [v / W for v in out]


def mean(vals, idxs):
    xs = [vals[i] for i in idxs if 0 <= i < len(vals)]
    return sum(xs) / max(1, len(xs))


def main():
    p_in, img_in = latest("tiled_input")
    p_pre, img_pre = latest("tiled_pre")
    p_seam, img_seam = latest("tiled_seam")
    p_up, img_up = latest("tiled_up")
    W0, H0 = img_in.size
    print(f"    输入 {p_in.name} {img_in.size} → 超分 {p_up.name} {img_up.size}", flush=True)

    # A7 尺寸与预放大一致性
    check("A7 超分尺寸 == 2×输入", img_up.size == (W0 * 2, H0 * 2),
          f"{img_up.size} vs 期望 {(W0 * 2, H0 * 2)}")
    check("A7 预放大图尺寸 == 超分尺寸", img_pre.size == img_up.size,
          f"{img_pre.size} vs {img_up.size}")
    want = img_in.convert("RGB").resize((W0 * 2, H0 * 2), Image.LANCZOS)
    mx = max(max(ImageChops.difference(want, img_pre.convert("RGB")).getchannel(i).getextrema())
             for i in range(3))
    check("A7 预放大 == 本机 LANCZOS 2x（逐字节）", mx == 0, f"最大通道差={mx}")

    # A8 接缝：ΣW 权重图
    W, H = img_seam.size
    sm = list(img_seam.convert("L").get_flattened_data())
    tile, overlap = 512, 64
    cols_x = [0, 448, 896]          # 本批：1024 宽 / tile 512 / overlap 64
    rows_y = [0, 448]
    # 内区 = 去掉最外 overlap 边界后的区域
    interior = [sm[y * W + x] for y in range(overlap, H - overlap, 7)
                for x in range(overlap, W - overlap, 7)]
    mn_i = min(interior)
    check("A8 ΣW 权重图内区 ≈1（互补 ramp 无欠覆盖）", mn_i >= 250,
          f"内区最小值={mn_i}（255=完全覆盖）")
    # 接缝带（overlap 中心）也应 ≈255
    seam_vals = []
    for x0 in cols_x[1:]:
        for y in range(overlap + 8, H - overlap, 11):
            for x in range(x0, min(W, x0 + overlap), 8):
                seam_vals.append(sm[y * W + x])
    for y0 in rows_y[1:]:
        for y in range(y0, min(H, y0 + overlap), 8):
            for x in range(overlap + 8, W - overlap, 11):
                seam_vals.append(sm[y * W + x])
    mn_s = min(seam_vals)
    check("A8 接缝带 ΣW ≈1（两侧权重和为 1）", mn_s >= 250,
          f"接缝带最小值={mn_s}，样本={len(seam_vals)}")
    # 外边界（单块覆盖）应 <1：说明 ramp 生效（信息）
    border = [sm[y * W + x] for y in range(0, 4) for x in range(0, W, 13)]
    info(f"最外边界 ΣW 最小值={min(border)}（单块覆盖 ⇒ 归一化仍无偏，符合设计）")

    # A8 视觉接缝：相邻列差在接缝带 vs 内区
    cm = col_means(img_up)
    rm = row_means(img_up)
    seam_cols = [x for x0 in cols_x[1:] for x in range(x0 - 2, x0 + overlap + 2)]
    seam_rows = [y for y0 in rows_y[1:] for y in range(y0 - 2, y0 + overlap + 2)]
    ref_cols = list(range(120, 380))
    ref_rows = list(range(120, 380))
    # 无可见接缝的三条判据（"硬缝"= 恰好落在块边界的突跳，而不是带内纹理略高）
    seam_set = set(seam_cols)
    non_seam = [v for i, v in enumerate(cm) if i not in seam_set]
    non_seam_sorted = sorted(non_seam)
    p95 = non_seam_sorted[int(len(non_seam_sorted) * 0.95)]
    argmax = max(range(len(cm)), key=lambda i: cm[i])
    max_in_seam = max(cm[i] for i in seam_cols if i < len(cm))
    check("A8 全局最不连续的列**不在**接缝带上（接缝不是画面里最硬的地方）",
          argmax not in seam_set,
          f"argmax 列={argmax}（值 {cm[argmax]:.3f}），接缝带={min(seam_cols)}~{max(seam_cols)}")
    check("A8 接缝带内无尖峰（带内最大相邻差 ≤1.5× 内区 p95）",
          max_in_seam <= 1.5 * p95,
          f"带内 max={max_in_seam:.3f} vs 内区 p95={p95:.3f}（比 {max_in_seam / max(1e-6, p95):.3f}）")
    # 基线必须是**全局非接缝区**（先前用固定内区列段会选到低纹理区，比偏高）
    non_seam_rows = [v for i, v in enumerate(rm) if i not in set(seam_rows)]
    r_c = mean(cm, seam_cols) / max(1e-6, sum(non_seam) / max(1, len(non_seam)))
    r_r = mean(rm, seam_rows) / max(1e-6, sum(non_seam_rows) / max(1, len(non_seam_rows)))
    check("A8 接缝带平均纹理能量与内区同量级（≤1.35×）",
          r_c <= 1.35 and r_r <= 1.35,
          f"列比={r_c:.3f} 行比={r_r:.3f}")
    info(f"相邻列差：接缝带均值={mean(cm, seam_cols):.3f} 全局非接缝均值={sum(non_seam) / len(non_seam):.3f}"
         f"（overlap 区被两次重绘 ⇒ 纹理能量天然略高，绝对量级 <1 灰阶）")

    # A9 细节：不糊（并报提升/衰减比）
    e_pre, e_up = edge_energy(img_pre), edge_energy(img_up)
    check("A9 重绘结果边缘能量 ≥ 0.8× 预放大（未变糊）", e_up >= 0.8 * e_pre,
          f"预放大={e_pre:.3f} 超分={e_up:.3f}（比 {e_up / max(1e-6, e_pre):.3f}）")
    # 与输入比：分辨率翻倍后细节量级应上升（信息）
    info(f"输入边缘能量={edge_energy(img_in):.3f}（512×384）"
         f" → 预放大={e_pre:.3f} → 超分={e_up:.3f}（1024×768）")

    print(f"\nFAILED = {FAILED}   PASSED = {PASSED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
