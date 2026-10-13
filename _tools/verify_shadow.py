#!/usr/bin/env python3
"""image-shadow 落地复核（P0-1 判据 N3/N4/N5/N7，**真实电商样例图**，纯 PIL 零 numpy）。

素材：`~/flowx-input/ecom/`（`fetch_samples.py` 抓的真实商品图 / 场景图，来源见 SOURCES.txt）。
本脚本直接调用节点真实代码路径（`nodes/image-shadow/main.py` 的 `process()`），
不碰推理服务、不需要 GPU —— 因此在手机本机就能跑出带数字的判据与物料。

判据：
  N3 投影框外**逐字节不变**（用节点输出的 shadow_box，不是近似框）
  N4 几何物理：投影质心偏移 == 位移向量（±1px）；压扁后影高 ≈ 原高×squash；
     blur 增大 ⇒ 投影覆盖像素数单调增
  N5 透明度：把 shadow_box 区域当画布，峰值 alpha ≈ round(opacity×255)（±2）
  N7 人眼物料：把「商品/影子/场景合成/参数扫描」拼成联系表写到 `~/flowx-view/`

用法：python3 nodes/_tools/verify_shadow.py
"""
import importlib.util
import pathlib
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
SAMPLES = pathlib.Path.home() / "flowx-input" / "ecom"
VIEW = pathlib.Path.home() / "flowx-view"
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


def load_mod():
    spec = importlib.util.spec_from_file_location(
        "shadow_main_v", ROOT / "image-shadow" / "main.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "image-shadow"))
    spec.loader.exec_module(mod)
    return mod


def pick(prefix, nth=0):
    files = sorted(SAMPLES.glob(prefix))
    if not files:
        raise SystemExit(f"缺素材：{SAMPLES}/{prefix}（先跑 fetch_samples.py）")
    return files[nth]


def centroid(alpha, bbox=None):
    px = alpha.load()
    x0, y0, x1, y1 = bbox or (0, 0, alpha.width, alpha.height)
    sx = sy = n = 0
    for y in range(y0, y1):
        for x in range(x0, x1):
            v = px[x, y]
            if v:
                sx += x * v
                sy += y * v
                n += v
    return (sx / n, sy / n, n) if n else (0.0, 0.0, 0)


def nonzero(alpha):
    return sum(alpha.histogram()[1:])


def outside_identical(orig_rgb, out_rgb, box):
    """投影框外逐字节比对（真实商品图）。两边都先转 RGB —— RGBA 的 4 元组与 RGB 的
    3 元组永不相等，不转换就是假失败。"""
    orig_rgb, out_rgb = orig_rgb.convert("RGB"), out_rgb.convert("RGB")
    a, b = orig_rgb.load(), out_rgb.load()
    x0, y0, x1, y1 = box
    bad = 0
    for y in range(min(orig_rgb.height, out_rgb.height)):
        for x in range(min(orig_rgb.width, out_rgb.width)):
            if x0 <= x < x1 and y0 <= y < y1:
                continue
            if a[x, y] != b[x, y]:
                bad += 1
    return bad


def e2e_check():
    """复核 Studio e2e 落盘的产物（~/flowx-e2e-out/ecom_bg_replace_shadow_*.png）。

    e2e 链路（wf156 `ecom-bg-replace-local`）：Ensure → LoadProduct(真实商品图) →
    LoadScene(真实场景图) → image-shadow(background=场景, canvas=expand) → save-image。

    判据：
      E1 输出是 expand 结果（尺寸 ≥ 商品尺寸，且向下变高）
      E2 主体全不透明像素逐字节 == 商品原图（贴在场景上不该改变不透明像素）
      E3 商品下方区域整体比场景同坐标更暗（投影真的画上去了）
    """
    out_dir = pathlib.Path.home() / "flowx-e2e-out"
    pngs = sorted(out_dir.glob("ecom_bg_replace_shadow_*.png"))
    if not pngs:
        print("SKIP e2e：~/flowx-e2e-out 下没有 ecom_bg_replace_shadow_*.png")
        return 0
    shot = pngs[-1]
    m = load_mod()
    product = Image.open(pick("prod_fakestore_*.png")).convert("RGBA")
    scene = Image.open(pick("bg_scene_*.jpg")).convert("RGB")
    got = Image.open(shot).convert("RGB")
    print(f"e2e 产物：{shot.name} {got.size}（商品 {product.size} / 场景 {scene.size}）")

    check("E1 e2e 输出是 expand 结果（≥ 商品尺寸且变高）",
          got.width >= product.width and got.height > product.height,
          f"{got.size} vs 商品 {product.size}")

    # angle=90 ⇒ dx=0，expand 只向下/向两侧按需扩；实测 subject_at=(58,0)
    ox, oy = (got.width - product.width) // 2, 0
    a_px, b_px = product.convert("RGB").load(), got.load()
    m_px = m.subject_alpha(product).load()
    bad = sum(1 for y in range(product.height) for x in range(product.width)
              if m_px[x, y] == 255 and b_px[x + ox, y + oy] != a_px[x, y])
    check("E2 e2e 主体全不透明像素逐字节 == 商品原图", bad == 0,
          f"subject_at=({ox},{oy}) 改动 {bad}px")

    # E3：商品下方（投影主要落区）整体更暗（与场景同坐标比）
    # ⚠️ 要比"显著差异"（±6/三通道），不能比"任意不等"：场景图经 服务端上传→下载
    # （JPEG 解码 → PNG 重编码）后±1 的舍入差会被算成"更亮"，判据就失真了（同 verify_ecom C4 口径）
    # ⚠️ 背景是被贴在**主体位置** (ox,oy) 的，且画布撑大后走"边缘延展"补边 ⇒
    #    对照底图必须同样按 (ox,oy) 偏移 + 边缘钳制取像素，否则比较的是错位的内容
    s_px = scene.load()
    rows = range(product.height, got.height)
    darker = lighter = 0
    for y in rows:
        for x in range(got.width):
            bx = min(max(x - ox, 0), scene.width - 1)
            by = min(max(y - oy, 0), scene.height - 1)
            d = sum(b_px[x, y]) - sum(s_px[bx, by])
            if d < -6:
                darker += 1
            elif d > 6:
                lighter += 1
    check("E3 e2e 商品下方区域整体变暗（投影画上去了）", darker > lighter * 3,
          f"更暗 {darker}px vs 更亮 {lighter}px（阈值±6/三通道）")

    # E4：产物与节点侧参数一致（尺寸公式：商品宽 = 输出宽 - 2×ox）
    check("E4 e2e 尺寸自洽（水平居中）", got.width - product.width == 2 * ox,
          f"W={got.width} w={product.width} ox={ox}")

    # E5 物料：整链四联（商品 / 场景 / 投影成品 / 加水印主图）+ 索引
    mains = sorted(out_dir.glob("ecom_bg_replace_main_*.png"))
    panels = [("真实商品图（FakeStore，透明底）", product),
              ("真实场景图（背景）", scene),
              (f"e2e 投影成品 exec 产物 {shot.name[-12:-4]}", got)]
    if mains:
        panels.append(("e2e 水印主图", Image.open(mains[-1])))
    sheet = VIEW / "103_shadow_e2e_chain.png"
    _grid([[p for _l, p in panels]], sheet, labels=[l for l, _p in panels])
    check("E5 e2e 链路物料落盘", sheet.is_file() and sheet.stat().st_size > 2000,
          sheet.name)
    _index_html([
        ("103_shadow_e2e_chain.png", "G11 投影：真实素材 e2e 链路（商品/场景/投影成品/水印主图）"),
        ("100_shadow_overview.png", "G11 投影：商品/影子/成品对照（真实电商图）"),
        ("101_shadow_sweep.png", "G11 投影：6 档参数扫描（距离/模糊/透明度/压扁/方向/淡出）"),
        ("102_shadow_other_product.png", "G11 投影：另一张真实商品图（webp+alpha）")])
    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


def main():
    if "--e2e" in sys.argv:
        return e2e_check()
    m = load_mod()
    product = pick("prod_fakestore_*.png")          # 真实电商商品图（透明背景）
    product2 = pick("prod_dummyjson_*.webp")        # 另一形态（webp+alpha）
    scene = pick("bg_scene_*.jpg")                  # 真实场景/背景图
    print(f"素材：商品={product.name} / {product2.name} / 场景={scene.name}")

    prod = Image.open(product).convert("RGBA")
    prod2 = Image.open(product2).convert("RGBA")
    bg = Image.open(scene).convert("RGB")
    print(f"尺寸：商品 {prod.size} / {prod2.size} / 场景 {bg.size}")

    # ---------- N4 几何：真实商品图上的位移/压扁/blur 单调 ----------
    # ⚠️ 真实商品图几乎铺满画布，直接位移会被画布边界裁掉（质心偏移量因此偏小，
    #    与节点行为无关）⇒ 几何量测必须在**加了 pad 的画布**上做。
    alpha0 = m.subject_alpha(prod)
    pad = 260
    alpha = Image.new("L", (prod.width + 2 * pad, prod.height + 2 * pad), 0)
    alpha.paste(alpha0, (pad, pad))
    bb = m.content_bbox(alpha)
    c0 = centroid(alpha, bb)
    for dist, ang in ((40, 90), (40, 0), (60, 180)):
        _, sh, off, _ = m.build_shadow(alpha, distance=dist, angle=ang, blur=0,
                                       opacity=1.0, squash=1.0, fade=0.0)
        c1 = centroid(sh)
        want = m.shadow_offset(dist, ang)
        check(f"N4 质心偏移 == 位移向量 d={dist} a={ang}",
              abs(c1[0] - c0[0] - want[0]) <= 1.0
              and abs(c1[1] - c0[1] - want[1]) <= 1.0,
              f"实得 ({c1[0]-c0[0]:.1f},{c1[1]-c0[1]:.1f}) 期望 {want}")

    h0 = bb[3] - bb[1]
    for sq in (0.3, 0.55, 1.0):
        _, sh, _, _ = m.build_shadow(alpha, distance=0, angle=90, blur=0,
                                     opacity=1.0, squash=sq, fade=0.0)
        bb2 = m.content_bbox(sh)
        want_h = max(1, round(h0 * sq))
        check(f"N4 压扁 {sq} ⇒ 影高 ≈ 原高×{sq}", abs((bb2[3] - bb2[1]) - want_h) <= 1,
              f"影高={bb2[3]-bb2[1]} 期望≈{want_h}")

    counts = []
    for b in (0, 6, 18, 36):
        _, sh, _, _ = m.build_shadow(alpha, distance=0, angle=90, blur=b,
                                     opacity=1.0, squash=1.0, fade=0.0)
        counts.append(nonzero(sh))
    check("N4 blur 单调增（0<6<18<36）",
          all(counts[i] < counts[i + 1] for i in range(len(counts) - 1)), f"{counts}")

    # ---------- N5 透明度（真实图上取 shadow_box 内的峰值） ----------
    for op, want in ((0.25, 64), (0.45, 115), (0.8, 204)):
        _, sh, _, peak = m.build_shadow(alpha, distance=20, angle=90, blur=6,
                                        opacity=op, squash=0.6, fade=0.0)
        check(f"N5 峰值 alpha ≈ round({op}×255)", abs(peak - want) <= 2,
              f"实得 {peak}（期望 {want}）")

    # ---------- N3 投影框外逐字节不变（真实商品 + 真实场景） ----------
    # ⚠️ 口径：先把商品合成到场景上，得到"已经是最终画面的底图"，再对它加投影。
    #    若把商品当独立前景 + 场景当 background 一起喂进去，主体本身就会改像素
    #    （那是合成的功劳/锅），"框外逐字节不变"这条判据就不成立。
    big_bg = bg.resize(prod.size, Image.LANCZOS)
    flat = Image.new("RGB", prod.size, (255, 255, 255))
    flat.paste(big_bg, (0, 0))
    flat.paste(prod.convert("RGB"), (0, 0), m.subject_alpha(prod))
    # 主体蒙版显式传（底图已含商品）；background 传同一张底图 ⇒ 只有投影会改像素。
    # ⚠️ 不传 background 的话输出是透明画布、不透明主体会把影子整个盖住（RGBA 路径语义）
    o1, box1, meta1 = m.process(flat, mask=m.subject_alpha(prod),
                                background=flat, distance=48, angle=90, blur=28,
                                opacity=0.5, squash=0.45, fade=0.25,
                                canvas="keep")
    bad = outside_identical(flat, o1, box1)
    check("N3 投影框外逐字节不变（真实商品+场景）", bad == 0, f"框外改动 {bad}px")
    check("N3 投影框内确有改动", o1.tobytes() != flat.tobytes())
    print(f"     shadow_box={box1} offset={meta1['offset']} peak={meta1['peak_alpha']}")

    # expand 模式：主体像素逐字节保留（透明底商品贴到场景上的合成路径）
    o2, box2, meta2 = m.process(prod, background=big_bg, distance=64, angle=90,
                                blur=20, opacity=0.55, squash=0.5, fade=0.1,
                                canvas="expand")
    ox, oy = meta2["subject_at"]
    a_px, b_px = prod.convert("RGB").load(), o2.convert("RGB").load()
    m_px = m.subject_alpha(prod).load()
    # ⚠️ 只比 alpha==255 的像素：半透明边缘是"与背景混合"的正常结果，不是没保留
    bad2 = sum(1 for y in range(prod.height) for x in range(prod.width)
               if m_px[x, y] == 255 and b_px[x + ox, y + oy] != a_px[x, y])
    check("N3 expand：主体全不透明像素逐字节保留", bad2 == 0,
          f"subject_at=({ox},{oy}) 改动 {bad2}px")

    # ---------- N7 物料：场景合成三联 + 参数扫描联系表 ----------
    VIEW.mkdir(exist_ok=True)
    tiles = [("商品原图（真实电商图）", _panel("", prod, 360)),
             ("投影（合成到白底看形状）", _panel("", _shadow_only(m, prod, 90), 360)),
             ("加投影的成品（真实场景底图）", _panel("", o1, 360)),
             ("商品②+投影（expand/另一张图）", _panel("", o2, 360))]

    sweep = []
    for label, kw in (("默认 24/24/0.45/0.55", dict()),
                      ("近影 12/8/0.6/0.7", dict(distance=12, blur=8, opacity=0.6, squash=0.7)),
                      ("长影 90/40/0.35/0.45", dict(distance=90, blur=40, opacity=0.35, squash=0.45)),
                      ("斜影 70@135°/30/0.5", dict(distance=70, angle=135, blur=30, opacity=0.5)),
                      ("无压扁 40/24/0.5/1.0", dict(distance=40, blur=24, opacity=0.5, squash=1.0)),
                      ("远端淡出 fade=0.8", dict(distance=70, blur=24, opacity=0.6, fade=0.8))):
        o, _b, _mt = m.process(prod, background=big_bg, canvas="expand", **kw)
        sweep.append((label, o))

    out3 = VIEW / "100_shadow_overview.png"
    _grid([[t[1] for t in tiles]], out3, labels=[t[0] for t in tiles])  # 4 格一行
    out4 = VIEW / "101_shadow_sweep.png"
    _grid([[s[1] for s in sweep[:3]], [s[1] for s in sweep[3:]]], out4,
          labels=[s[0] for s in sweep])  # sweep 元素是 (label, image)
    # 换一张商品图（webp+alpha）与不同背景，验证不是只对一张图成立
    o5, box5, _ = m.process(prod2, background=bg.resize(prod2.size),
                            distance=40, angle=90, blur=18, opacity=0.5,
                            squash=0.5, canvas="keep")
    out5 = VIEW / "102_shadow_other_product.png"
    _grid([[_panel("商品②（webp+alpha）", prod2.resize((360, 360)), 360),
            _panel("商品②+投影（真实场景）", o5, 360)]], out5)
    check("N7 物料落盘（3 张联系表）",
          all(p.is_file() and p.stat().st_size > 2000 for p in (out3, out4, out5)),
          f"{out3.name} {out4.name} {out5.name}")
    _material_selfcheck(flat, o1, box1, sweep, tiles[1][1],
                        (out3, out4, out5))
    print(f"     物料：{out3}\n           {out4}\n           {out5}")
    _index_html()

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


def _material_selfcheck(flat, composed, box, sweep, shadow_panel, sheets):
    """物料自检（替代"抽图看图"的客观代理 —— 验收环境看不到图时用）。

    ① 影子真的出现在成品里（成品影区比原底图暗）
    ② 参数扫描每一格互不相同（不是同一张图贴了 6 个标题）
    ③ "只有影子"面板以白底为主且有明显暗区（不是黑块 / 不是空白）
    ④ 三张联系表都存在、可解码、尺寸非退化
    """
    # ① 成品影区（shadow_box 的下 2/3）必须整体比原底图暗
    x0, y0, x1, y1 = box
    a, b = flat.convert("RGB").load(), composed.convert("RGB").load()
    darker = lighter = 0
    for y in range(y0 + (y1 - y0) // 3, y1):
        for x in range(x0, x1):
            sa, sb = sum(a[x, y]), sum(b[x, y])
            darker += sb < sa
            lighter += sb > sa
    check("物料① 成品里影子确实压暗了底图", darker > lighter * 3,
          f"更暗 {darker}px vs 更亮 {lighter}px")

    # ② 扫描格两两不同
    import hashlib
    digests = [hashlib.sha256(im.tobytes()).hexdigest() for _l, im in sweep]
    check("物料② 6 档参数产出互不相同", len(set(digests)) == len(digests),
          f"{len(set(digests))}/{len(digests)} 唯一")

    # ③ 影子面板：白底为主 + 存在暗区（能看出形状）
    g = shadow_panel.convert("L")
    hist = g.histogram()
    total = sum(hist)
    bright = sum(hist[240:]) / total
    dark = sum(hist[:200]) / total
    check("物料③ 影子面板白底为主且有暗区", bright > 0.5 and dark > 0.02,
          f"高亮={bright:.1%} 暗区={dark:.1%}")

    # ④ 联系表可解码且非退化
    for pth in sheets:
        im = Image.open(pth).convert("RGB")
        bins = sum(1 for v in im.convert("L").histogram() if v)  # 非零灰度桶
        check(f"物料④ {pth.name} 可解码且非纯色",
              im.width > 400 and bins > 20, f"{im.size} 灰度桶={bins}")


def _index_html(items=None):
    """把本次物料追加进 ~/flowx-view/index.html（8899 索引，供人眼验收）。"""
    idx = VIEW / "index.html"
    if not idx.is_file():
        return
    html = idx.read_text()
    items = items or [
        ("100_shadow_overview.png", "G11 投影：商品/影子/成品对照（真实电商图）"),
        ("101_shadow_sweep.png", "G11 投影：6 档参数扫描（距离/模糊/透明度/压扁/方向/淡出）"),
        ("102_shadow_other_product.png", "G11 投影：另一张真实商品图（webp+alpha）")]
    add = "".join(f'<li><a href="{f}">🆕 {t}</a></li>' for f, t in items
                  if f not in html)
    if not add:
        return
    html = html.replace("<hr>", f"{add}</ul><hr>", 1) if "</ul><hr>" not in html \
        else html
    # 顶部列表：插在 </ul> 之前；底部图区：追加 <h3>/<img>
    if add:
        html = html.replace("</ul>", add + "</ul>", 1)
        html = html.replace(
            "</body>",
            "".join(f"<h3>{t}</h3><img src='{f}'>" for f, t in items
                    if f"src='{f}'" not in html) + "</body>")
    idx.write_text(html)
    print(f"     索引已更新：{idx}")


def _shadow_only(m, prod, angle, pad=120):
    """**只有影子层**（不含主体）合成到白底：人眼确认形状/压扁/淡出。

    ⚠️ 不能用 process(无 background)：那条路径会把主体也画上（RGBA 画布 = 影子+主体），
    面板里就一半是商品本身了 —— 标签必须与内容一致。
    """
    a0 = m.subject_alpha(prod)
    big = Image.new("L", (prod.width + 2 * pad, prod.height + 2 * pad), 0)
    big.paste(a0, (pad, pad))
    shadow_rgba, _sa, _off, _pk = m.build_shadow(
        big, distance=48, angle=angle, blur=24, opacity=0.6, squash=0.45, fade=0.2)
    white = Image.new("RGBA", shadow_rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, shadow_rgba).convert("RGB")


def _panel(label, im, size):
    """统一缩放到 size×size 的白/格底面板（人眼物料用）。"""
    im = im.convert("RGB")
    im = im.copy()
    im.thumbnail((size, size), Image.LANCZOS)
    bg = Image.new("RGB", (size, size), (245, 245, 247))
    bg.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return bg


def _grid(rows, path, labels=None, pad=8, header=26):
    """拼联系表（行 × 列），标签写在每格上方。"""
    from PIL import ImageDraw
    cols = max(len(r) for r in rows)
    cw = max(im.width for r in rows for im in r)
    ch = max(im.height for r in rows for im in r) + header
    W = pad + cols * (cw + pad)
    H = pad + len(rows) * (ch + pad)
    sheet = Image.new("RGB", (W, H), (28, 30, 36))
    d = ImageDraw.Draw(sheet)
    i = 0
    for r, row in enumerate(rows):
        for c, im in enumerate(row):
            x = pad + c * (cw + pad)
            y = pad + r * (ch + pad)
            if labels and i < len(labels):
                d.text((x + 4, y + 6), str(labels[i])[:44], fill=(230, 232, 240))
            sheet.paste(im, (x + (cw - im.width) // 2, y + header))
            i += 1
    sheet.save(path)
    return path


if __name__ == "__main__":
    sys.exit(main())
