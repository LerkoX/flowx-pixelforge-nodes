"""B2 / G1 去白边专项复核（dev-plan §48.11 补测）。

输入 = 白底商品图（主体贴在纯白上 ⇒ 边缘像素就是"主体 × 白底"混色）。
同一张图跑两遍 BiRefNet，只有 `refine` 不同（alpha 完全一致）：
  refine=off → RGBA 的 RGB 是原图（带白边）；refine=on → 官方前后景分离反解前景色（去白边）。
再各自换到纯绿底（#00B050），白边残留会被放大成浅色环。

判据：
  C1 对照干净：两遍 alpha 蒙版逐字节一致（只差 RGB）
  C2 输入确为白底场景（背景区近纯白）
  C3 前景色本身：过渡带 min(R,G,B) 与近白占比下降
  C4 换绿底后：过渡带 min(R,G,B) 与近白占比下降，绿色污染不升高
  C5 主体内部（alpha=255 核）几乎不变（不是整体调色）
  C6 绿底合成 == alpha 混合公式重算（≤1/255）
另出 3× 放大人眼对比拼图 defringe_preview.png。
"""
import pathlib
import re
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter

OUT = pathlib.Path.home() / "flowx-e2e-out"
FAILED = 0
PASSED = 0
LO, HI = 16, 239          # 过渡带 alpha 区间
WHITISH = 200             # "近白"阈值


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


def latest(prefix, mode="RGB"):
    pat = re.compile(rf"^{re.escape(prefix)}_+(\d{{8}}_\d{{6}})\.png$")
    fs = sorted([p for p in OUT.iterdir() if pat.match(p.name)], key=lambda p: p.stat().st_mtime)
    if not fs:
        sys.exit(f"找不到产物：{prefix}_*")
    im = Image.open(fs[-1])
    print(f"    使用 {fs[-1].name}")
    return im.convert(mode)


def band_metrics(rgb_img, alpha, box=None):
    """过渡带统计 → (像素数, 平均 min(R,G,B), 近白占比%, 平均绿污染)。box=(x0,y0,x1,y1) 可裁剪。"""
    if box:
        rgb_img = rgb_img.crop(box)
        alpha = alpha.crop(box)
    R, G, B = (c.get_flattened_data() for c in rgb_img.split())
    A = alpha.get_flattened_data()
    n = mn_sum = wh = grn_sum = 0
    for r, g, b, a in zip(R, G, B, A):
        if LO <= a <= HI:
            n += 1
            m = r if r < g else g
            if b < m:
                m = b
            mx = r if r > g else g
            if b > mx:
                mx = b
            mn_sum += m
            if m >= WHITISH:
                wh += 1
            grn_sum += g - mx
    if not n:
        return 0, 0.0, 0.0, 0.0
    return n, mn_sum / n, wh / n * 100, grn_sum / n


def main():
    mask = latest("defringe_mask_plain", "L")
    mask_r = latest("defringe_mask_refined", "L")
    rgba_p = latest("defringe_rgba_plain", "RGBA")
    rgba_r = latest("defringe_rgba_refined", "RGBA")
    comp_p = latest("defringe_comp_plain")
    comp_r = latest("defringe_comp_refined")
    src = latest("defringe_input_whitebg")

    W, H = mask.size
    tot = W * H

    # C1 对照干净
    check("C1 对照干净：两遍 alpha 蒙版逐字节一致（只差 RGB）",
          mask.tobytes() == mask_r.tobytes(),
          f"前景占比={sum(1 for v in mask.get_flattened_data() if v > 127) / tot * 100:.1f}%")

    # C2 白底场景：只看**远离轮廓**的背景（白底图由"扩张+羽化"后的精修蒙版生成，
    # 紧贴轮廓的一圈会落在原蒙版 alpha=0 区内，故先把背景区腐蚀掉边界 4px 再判）
    bgmask = mask.point(lambda v: 255 if v == 0 else 0).filter(ImageFilter.MinFilter(9))
    A = bgmask.get_flattened_data()
    R, G, B = (c.get_flattened_data() for c in src.split())
    nbg = nbw = 0
    s = 0.0
    worst = 0
    for i, a in enumerate(A):
        if a:
            nbg += 1
            s += (R[i] + G[i] + B[i]) / 3
            if R[i] == 255 and G[i] == 255 and B[i] == 255:
                nbw += 1
            else:
                worst = max(worst, 255 - min(R[i], G[i], B[i]))
    check("C2 输入确为白底场景（远离轮廓的背景区逐像素纯白）",
          nbg and nbw / nbg >= 0.999,
          f"纯白占比={nbw / max(1, nbg) * 100:.3f}%（{nbg}px）均值={s / max(1, nbg):.3f} 最大偏差={worst}")

    # C3 前景色本身
    n, mp, wp, _ = band_metrics(rgba_p.convert("RGB"), mask)
    _, mr, wr, _ = band_metrics(rgba_r.convert("RGB"), mask)
    check("C3 过渡带前景色变不白（min(R,G,B) 下降）", mr < mp - 1.0,
          f"refine 关={mp:.1f} 开={mr:.1f}（Δ={mp - mr:+.1f}）")
    check("C3 过渡带近白像素占比不升", wr <= wp + 0.05, f"关={wp:.1f}% 开={wr:.1f}%（Δ={wp - wr:+.1f}）")
    info(f"过渡带 {n}px（占全图 {n / tot * 100:.2f}%）")

    # C4 绿底合成
    _, cp, wp2, gp = band_metrics(comp_p.convert("RGB"), mask)
    _, cr, wr2, gr = band_metrics(comp_r.convert("RGB"), mask)
    check("C4 绿底合成：过渡带白边残留（min(R,G,B)）下降", cr < cp - 1.0,
          f"关={cp:.1f} 开={cr:.1f}（Δ={cp - cr:+.1f}）")
    check("C4 绿底合成：近白占比不升且绿色污染不升",
          wr2 <= wp2 + 0.05 and gr <= gp + 1.0,
          f"近白 关={wp2:.1f}% 开={wr2:.1f}% | 绿污染 关={gp:.1f} 开={gr:.1f}")

    # C5 内部不变
    core = mask.point(lambda v: 255 if v == 255 else 0).filter(ImageFilter.MinFilter(7))
    d = ImageChops.difference(rgba_p.convert("RGB"), rgba_r.convert("RGB")).convert("L")
    cA = core.get_flattened_data()
    dA = d.get_flattened_data()
    nc = sum(1 for v in cA if v)
    interior = sum(dA[i] for i, v in enumerate(cA) if v) / max(1, nc)
    check("C5 主体内部（alpha=255 核）几乎不变（非整体调色）", interior < 3.0,
          f"核内 {nc}px 平均差={interior:.2f}/255")

    # C6 合成公式
    for tag, rgba, comp in (("refine 关", rgba_p, comp_p), ("refine 开", rgba_r, comp_r)):
        want = Image.composite(rgba.convert("RGB"), Image.new("RGB", rgba.size, (0, 176, 80)), rgba.split()[3])
        mx = max(max(ImageChops.difference(want, comp.convert("RGB")).getchannel(i).getextrema()) for i in range(3))
        check(f"C6 绿底合成 == alpha 混合公式（{tag}，≤1/255）", mx <= 1, f"最大通道差={mx}")

    # 人眼预览：过渡带最密处 3× 放大并排
    dens = mask.point(lambda v: 255 if LO <= v <= HI else 0).resize((W // 12, H // 12))
    px = dens.load()
    bx, by = max(((x, y) for y in range(dens.size[1]) for x in range(dens.size[0])), key=lambda p: px[p])
    x0 = max(0, min(W - 200, bx * 12 - 50))
    y0 = max(0, min(H - 200, by * 12 - 50))
    box = (x0, y0, x0 + 200, y0 + 200)
    Z, tile = 3, 600
    canvas = Image.new("RGB", (tile * 2 + 36, tile + 44), (28, 28, 30))
    for i, (comp, label) in enumerate(((comp_p, "refine=off (white fringe)"), (comp_r, "refine=on (defringed)"))):
        canvas.paste(comp.convert("RGB").crop(box).resize((tile, tile), Image.NEAREST), (12 + i * (tile + 12), 34))
        ImageDraw.Draw(canvas).text((16 + i * (tile + 12), 12), label, fill=(240, 240, 240))
    prev = OUT / "defringe_preview.png"
    canvas.save(prev)
    info(f"人眼预览（3× 放大，{box}）：{prev}")

    print(f"\nFAILED = {FAILED}   PASSED = {PASSED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
