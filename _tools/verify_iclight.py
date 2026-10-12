#!/usr/bin/env python3
"""B4/S3 G9 IC-Light 重打光落盘复核（dev-plan §50.5 判据 A1~A7）。

只读 `~/flowx-e2e-out/` 的 PNG，纯 PIL + 标准库（Termux 无 numpy），不碰服务。

判据（阈值按实测标定，见 dev-plan §50.11）：
  A1 尺寸：left/right == 输入尺寸；hi == 1.5× 输入且对齐 64
  A2 确定性：同参数复跑（icl_left vs icl_rep）**逐字节一致**
  A3 光方向物理判据：主体区（mask>128）左/右半平均亮度差 —— left 显著为正、
     right 显著为负、且 Δleft-Δright ≥ 60
  A4 方向由 light_dir 主导（相对输入的方向位移）：Δleft−Δin ≥ +30 且 Δin−Δright ≥ +30
     （输入图自身可能不均匀，所以判"相对输入移动了多少"，而不是"输入是否均匀"）
  A5 主体保留（重打光会改写低频光照，逐像素/边缘相关**不是**正确判据——实测仅 0.15~0.30，
     仅作观察值打印；正确判据是"对输出再抠一次图，与输入蒙版比 IoU"≥ 0.90，实测 0.9823）
  A6 highres 阶段确实生效：尺寸 1.5× 且与低清输出逐字节不同
  A7 执行器无关（有 docker 产物时）：icld_* 与 icl_* **逐字节一致**
"""
import hashlib
import pathlib
import re
import sys

from PIL import Image, ImageFilter

OUT = pathlib.Path.home() / "flowx-e2e-out"
FAILED = 0
PASSED = 0


def check(label, cond, extra=""):
    global FAILED, PASSED
    if cond:
        PASSED += 1
        print(f"PASS {label}" + (f"  {extra}" if extra else ""))
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  {extra}" if extra else ""))


def latest(prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_+\d{{8}}_\d{{6}}\.png$")
    fs = sorted([f for f in OUT.iterdir() if pat.match(f.name)],
                key=lambda f: f.stat().st_mtime)
    return fs[-1] if fs else None


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lum_balance(img, mask, half=None):
    """主体区（mask>128）左/右半平均亮度 + 平衡差。纯 Python，无 numpy。"""
    g = img.convert("L")
    gpx = g.load()
    mpx = mask.convert("L").load() if mask is not None else None
    half = g.width // 2
    sl = sr = nl = nr = 0
    for y in range(g.height):
        for x in range(g.width):
            if mpx is not None and mpx[x, y] <= 128:
                continue
            v = gpx[x, y]
            if x < half:
                sl += v
                nl += 1
            else:
                sr += v
                nr += 1
    lm = sl / nl if nl else 0.0
    rm = sr / nr if nr else 0.0
    return lm, rm, lm - rm


def edge_corr(a, b, mask):
    """主体区边缘图皮尔逊相关（Sobel 近似：FIND_EDGES + 主体掩膜）。"""
    ea = a.convert("L").filter(ImageFilter.FIND_EDGES)
    eb = b.convert("L").filter(ImageFilter.FIND_EDGES)
    pa, pb = ea.load(), eb.load()
    mp = mask.convert("L").load()
    n = sa = sb = saa = sbb = sab = 0
    for y in range(1, ea.height - 1):
        for x in range(1, ea.width - 1):
            if mp[x, y] <= 128:
                continue
            va, vb = pa[x, y], pb[x, y]
            n += 1
            sa += va
            sb += vb
            saa += va * va
            sbb += vb * vb
            sab += va * vb
    if n < 100:
        return 0.0
    ma, mb = sa / n, sb / n
    cov = sab / n - ma * mb
    va = saa / n - ma * ma
    vb = sbb / n - mb * mb
    if va <= 0 or vb <= 0:
        return 0.0
    return cov / (va ** 0.5 * vb ** 0.5)


def main():
    src, mask = latest("icl_input"), latest("icl_mask")
    left, right, rep, hi = (latest("icl_left"), latest("icl_right"),
                            latest("icl_rep"), latest("icl_hi"))
    if not all((src, mask, left, right, rep, hi)):
        print("缺产物：", {k: (v.name if v else None) for k, v in
                           (("input", src), ("mask", mask), ("left", left),
                            ("right", right), ("rep", rep), ("hi", hi))})
        return 1
    S = Image.open(src).convert("RGB")
    M = Image.open(mask).convert("L")
    L = Image.open(left).convert("RGB")
    R = Image.open(right).convert("RGB")
    P = Image.open(rep).convert("RGB")
    H = Image.open(hi).convert("RGB")
    print(f"输入 {S.size}  蒙版 {M.size}  left {L.size}  right {R.size}  "
          f"rep {P.size}  hi {H.size}")

    print("\n== A1 尺寸 ==")
    check("left 尺寸 == 输入尺寸", L.size == S.size, f"{L.size} vs {S.size}")
    check("right 尺寸 == 输入尺寸", R.size == S.size, f"{R.size} vs {S.size}")
    exp = (int(round(S.width * 1.5 / 64) * 64), int(round(S.height * 1.5 / 64) * 64))
    check(f"highres 尺寸 == 1.5× 输入且对齐 64（{exp}）", H.size == exp,
          f"{H.size} vs {exp}")

    print("\n== A2 确定性（同参数复跑逐字节） ==")
    check("icl_left == icl_rep（sha256）", sha(left) == sha(rep),
          f"{sha(left)[:12]} vs {sha(rep)[:12]}")

    print("\n== A3 光方向物理判据（主体区左右半亮度） ==")
    lin, rin, din = lum_balance(S, M)
    ll, rl, dl = lum_balance(L, M)
    lr, rr, dr = lum_balance(R, M)
    print(f"  输入  左 {lin:6.1f} / 右 {rin:6.1f}  Δ={din:+7.1f}")
    print(f"  left  左 {ll:6.1f} / 右 {rl:6.1f}  Δ={dl:+7.1f}")
    print(f"  right 左 {lr:6.1f} / 右 {rr:6.1f}  Δ={dr:+7.1f}")
    check("left 光：主体左侧显著更亮（Δ ≥ +30）", dl >= 30.0, f"Δ={dl:+.1f}")
    check("right 光：主体右侧显著更亮（Δ ≤ −30）", dr <= -30.0, f"Δ={dr:+.1f}")
    check("方向翻转幅度 ≥ 60", (dl - dr) >= 60.0, f"Δleft-Δright={dl - dr:+.1f}")

    print("\n== A4 方向由 light_dir 主导（相对输入） ==")
    check("left 相对输入左移 ≥ +30", (dl - din) >= 30.0, f"Δleft−Δin={dl - din:+.1f}")
    check("right 相对输入右移 ≥ +30", (din - dr) >= 30.0, f"Δin−Δright={din - dr:+.1f}")

    print("\n== A5 主体保留 ==")
    # 观察值（不作判据）：重打光改写低频光照 ⇒ 逐像素/边缘相关天然低
    c_in_left = edge_corr(S, L, M)
    c_in_right = edge_corr(S, R, M)
    print(f"  [观察] 主体区边缘相关（不作判据，仅参考）：left r={c_in_left:.3f} "
          f"right r={c_in_right:.3f}")
    matte = latest("icl_left_matte")
    if matte is None:
        check("需要 icl_left_matte（对输出再抠图）产物", False, "缺产物")
    else:
        B = Image.open(matte).convert("L")
        if B.size != M.size:
            B = B.resize(M.size, Image.NEAREST)
        pa, pb = M.load(), B.load()
        inter = union = 0
        for y in range(M.height):
            for x in range(M.width):
                a, b = pa[x, y] > 128, pb[x, y] > 128
                if a and b:
                    inter += 1
                if a or b:
                    union += 1
        iou = inter / union if union else 0.0
        check("对输出再抠图 vs 输入蒙版 IoU ≥ 0.90（主体轮廓保留）", iou >= 0.90,
              f"IoU={iou:.4f}")

    print("\n== A6 highres 阶段确实生效 ==")
    check("highres 输出与低清输出不同（逐字节）", sha(hi) != sha(left),
          f"{sha(hi)[:12]} vs {sha(left)[:12]}")

    print("\n== A7 执行器无关（docker 产物，可选） ==")
    pairs = [("icld_input", "icl_input"), ("icld_mask", "icl_mask"),
             ("icld_left", "icl_left"), ("icld_right", "icl_right"),
             ("icld_rep", "icl_rep"), ("icld_hi", "icl_hi")]
    found = 0
    for dp, lp in pairs:
        d, l = latest(dp), latest(lp)
        if d is None:
            continue
        found += 1
        check(f"{dp} == {lp}（逐字节）", sha(d) == sha(l),
              f"{sha(d)[:12]} vs {sha(l)[:12]}")
    if found == 0:
        print("  （尚无 docker 产物：跑 wf iclight-e2e 后再复核）")

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
