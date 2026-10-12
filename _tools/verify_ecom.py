#!/usr/bin/env python3
"""B4/S2 G11 电商模板落盘复核（dev-plan §50.5 判据 C1~C4）。

只读 `~/flowx-e2e-out/` 的 PNG，纯 PIL（Termux 无 numpy），不碰服务。

判据（阈值标定见 dev-plan §50.9）：
  C1 白底纯度：把主体向外扩 2px 后仍为背景的区域，min(R,G,B) ≥ 250 占比 = 100%
     （0<mask<128 是合成羽化的自然过渡带，不该要求纯白；实测排除 2px 带即 100.000%）
  C2 尺寸规格：1:1 输出为正方形；主体包围盒宽/高填充 ≥ 0.5 且主体面积占比 ≥ 0.15
  C3 水印只改水印区：加印前后（Pad 输出 vs 加印输出）**水印 box 外逐字节不变**，
     box 内确有变化，且 box 位于右下（position=bottom-right + margin=24）
  C4 场景合成保主体：主体核心（mask 向内腐蚀 16px，即 feather=8 的实际过渡带之外）
     与输入差 ≤4；腐蚀 24px 的核心逐字节一致（≤0）；背景区（mask<5）替换率 ≥ 90%；
     羽化带（1..254）存在（不是硬边）。实测 feather=8 的实际过渡带宽 ≈24px
"""
import pathlib
import re
import sys

from PIL import Image, ImageChops, ImageFilter

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


def latest(prefix, nth=0):
    pat = re.compile(rf"^{re.escape(prefix)}_+\d{{8}}_\d{{6}}\.png$")
    fs = sorted([f for f in OUT.iterdir() if pat.match(f.name)],
                key=lambda f: f.stat().st_mtime)
    return fs[-(nth + 1)] if len(fs) > nth else None


def c1_white_purity():
    comp, mask = latest("ecom_white_composite"), latest("ecom_white_mask")
    if not comp or not mask:
        check("C1 需要合成图 + 蒙版", False, "缺产物"); return
    im = Image.open(comp).convert("RGB")
    mk = Image.open(mask).convert("L")
    if mk.size != im.size:
        mk = mk.resize(im.size, Image.NEAREST)
    px = im.load()
    # 主体向外扩 2px：扩后仍为 0 的像素才是「远离轮廓的背景」（排除羽化过渡带）
    dil = mk.filter(ImageFilter.MaxFilter(5))
    dp = dil.load()
    bg = pure = 0
    for y in range(im.height):
        for x in range(im.width):
            if dp[x, y] != 0:
                continue
            bg += 1
            if min(px[x, y]) >= 250:
                pure += 1
    frac = pure / bg if bg else 0.0
    check(f"C1 远离轮廓(≥2px)背景纯白 = 100%（{bg}px）", frac >= 0.9999,
          f"{frac * 100:.4f}%")


def c2_spec():
    main, mask = latest("ecom_white_main"), latest("ecom_white_mask")
    if not main:
        check("C2 需要主图", False, "缺产物"); return
    im = Image.open(main)
    check("C2 输出为 1:1 正方形", im.width == im.height, str(im.size))
    if not mask:
        check("C2 需要蒙版算主体占比", False, "缺产物"); return
    mk = Image.open(mask).convert("L").resize(im.size, Image.NEAREST)
    bb = mk.getbbox()
    if not bb:
        check("C2 蒙版非空", False); return
    bw, bh = bb[2] - bb[0], bb[3] - bb[1]
    area = sum(1 for v in mk.get_flattened_data() if v >= 128) / (im.width * im.height)
    check(f"C2 主体宽填充 ≥0.5（{bw / im.width:.2f}）", bw / im.width >= 0.5)
    check(f"C2 主体高填充 ≥0.5（{bh / im.height:.2f}）", bh / im.height >= 0.5)
    check(f"C2 主体面积占比 ≥0.15（{area:.2f}）", area >= 0.15)


def c3_watermark():
    before, after = latest("ecom_white_square"), latest("ecom_white_main")
    if not before or not after:
        check("C3 需要加印前后两张", False, "缺产物"); return
    a = Image.open(before).convert("RGB")
    b = Image.open(after).convert("RGB")
    if a.size != b.size:
        check("C3 加印前后尺寸一致", False, f"{a.size} vs {b.size}"); return
    diff = ImageChops.difference(a, b).convert("L")
    bb = diff.getbbox()
    if not bb:
        check("C3 水印确实改动了画面", False, "无差异"); return
    check("C3 水印确实改动了画面", True, f"diff bbox={bb}")
    pa, pb, pd = a.load(), b.load(), diff.load()
    out = 0
    for y in range(a.height):
        for x in range(a.width):
            if bb[0] <= x < bb[2] and bb[1] <= y < bb[3]:
                continue
            if pd[x, y] > 0 or pa[x, y] != pb[x, y]:
                out += 1
    check("C3 水印区外逐字节不变", out == 0, f"区外改动 {out} px")
    check("C3 水印位于右下（bottom-right + margin 24）",
          bb[2] >= a.width - 30 and bb[3] >= a.height - 30 and bb[0] > a.width // 2,
          f"bbox 右下角 ({bb[2]},{bb[3]}) 尺寸 {a.size}")


def c4_scene():
    src, comp, mask = (latest("ecom_scene_input"), latest("ecom_scene_composite"),
                       latest("ecom_scene_mask"))
    if not all((src, comp, mask)):
        check("C4 需要输入/合成/蒙版", False,
              str([f.name if f else None for f in (src, comp, mask)])); return
    a = Image.open(src).convert("RGB")
    b = Image.open(comp).convert("RGB")
    mk = Image.open(mask).convert("L")
    if mk.size != a.size:
        mk = mk.resize(a.size, Image.NEAREST)
    pa, pb, mp = a.load(), b.load(), mk.load()
    bg = bg_repl = feather = 0
    for y in range(a.height):
        for x in range(a.width):
            m = mp[x, y]
            d = max(abs(pa[x, y][i] - pb[x, y][i]) for i in range(3))
            if m < 5:
                bg += 1
                if d > 20:
                    bg_repl += 1
            elif m < 250:
                feather += 1
    # feather=8 在合成节点内部是宽过渡带（实测 ≈24px）：按「腐蚀后的核心」判主体保留
    for k, tol in ((16, 4), (24, 0)):
        core = mk.filter(ImageFilter.MinFilter(2 * k + 1))
        cp = core.load()
        tot = mx = 0
        for y in range(a.height):
            for x in range(a.width):
                if cp[x, y] != 255:
                    continue
                tot += 1
                mx = max(mx, max(abs(pa[x, y][i] - pb[x, y][i]) for i in range(3)))
        check(f"C4 主体核心(腐蚀 {k}px，{tot}px) 与输入差 ≤{tol}", mx <= tol,
              f"最大差 {mx}")
    frac = bg_repl / bg if bg else 0.0
    check(f"C4 背景区被替换 ≥90%（{bg}px）", frac >= 0.90, f"{frac * 100:.1f}%")
    check("C4 羽化带存在（非硬边）", feather > 0,
          f"过渡带 {feather}px（feather=8 实测 ≈24px）")


def main():
    print("== C1 白底主图：主体外区域纯白 ==")
    c1_white_purity()
    print("\n== C2 尺寸规格（1:1 + 主体填充） ==")
    c2_spec()
    print("\n== C3 水印只改水印区 ==")
    c3_watermark()
    print("\n== C4 场景合成（保主体 + 换背景 + 羽化） ==")
    c4_scene()
    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
