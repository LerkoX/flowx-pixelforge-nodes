#!/usr/bin/env python3
"""image-shadow 本机单测（纯 PIL，零 numpy；不需要 GPU/推理服务）。

对应 dev-plan P0-1 的判据 N1~N6：
  N1 本机单测（本文件）      N2 逐字节确定性      N3 非投影区逐字节不变
  N4 几何物理判据（质心偏移/压扁高度/blur 单调）
  N5 透明度（峰值 alpha ≈ opacity）
  N6 画布模式（keep/expand + 主体区逐字节不变）

用法：python3 nodes/image-shadow/test_main.py
"""
import importlib.util
import pathlib
import sys

from PIL import Image

HERE = pathlib.Path(__file__).parent
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
    sys.path.insert(0, str(HERE))
    spec = importlib.util.spec_from_file_location("shadow_main", HERE / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_mask(size=(600, 800), box=(100, 100, 300, 400)):
    """主体蒙版：矩形实心（便于几何判据精确计算）。"""
    m = Image.new("L", size, 0)
    m.paste(Image.new("L", (box[2] - box[0], box[3] - box[1]), 255), (box[0], box[1]))
    return m, box


def centroid(alpha):
    """float 质心（用于几何判据，避免整型取整误差）。"""
    px = alpha.load()
    sx = sy = n = 0
    for y in range(alpha.height):
        for x in range(alpha.width):
            v = px[x, y]
            if v:
                sx += x * v
                sy += y * v
                n += v
    return (sx / n, sy / n) if n else (0.0, 0.0)


def nonzero(alpha):
    h = alpha.histogram()
    return sum(h[1:])


def outside_identical(orig, new, box, label):
    """投影框之外必须逐字节不变（N3）。box 为 None 时要求整幅相同。"""
    if box is None:
        check(f"{label} 无投影 ⇒ 输出逐字节等于输入",
              orig.convert("RGB").tobytes() == new.convert("RGB").tobytes())
        return
    ox, oy = orig.convert("RGB").load(), new.convert("RGB").load()
    x0, y0, x1, y1 = box
    diff = 0
    for y in range(min(orig.height, new.height)):
        for x in range(min(orig.width, new.width)):
            if x0 <= x < x1 and y0 <= y < y1:
                continue
            if ox[x, y] != oy[x, y]:
                diff += 1
    check(f"{label} 投影框外逐字节不变", diff == 0, f"框外改动 {diff} px")


def main():
    m = load_mod()

    # ---------- 参数解析 ----------
    check("颜色 #RRGGBB", m.parse_color("#112233") == (17, 34, 51))
    check("颜色 #RRGGBBAA 取前三通道", m.parse_color("#11223380") == (17, 34, 51))
    check("颜色 r,g,b", m.parse_color("10,20,30") == (10, 20, 30))
    check("颜色空值用默认", m.parse_color("", (1, 2, 3)) == (1, 2, 3))
    for bad in ("#FFF", "1,2", "zz", "#GGGGGG"):
        try:
            m.parse_color(bad)
            check(f"非法颜色报错 {bad!r}", False, "未报错")
        except ValueError:
            check(f"非法颜色报错 {bad!r}", True)

    # ---------- N4 位移向量（角度约定：0=+x，90=+y 向下） ----------
    for ang, want in ((0, (40, 0)), (90, (0, 40)), (180, (-40, 0)),
                      (270, (0, -40))):
        got = m.shadow_offset(40, ang)
        check(f"位移向量 angle={ang} ⇒ {want}", got == want, f"实得 {got}")

    # ---------- 主体 alpha 取法 ----------
    mask, _ = make_mask()
    rgb = Image.new("RGB", (600, 800), (10, 20, 30))
    check("无 alpha 的 RGB ⇒ 整图都是主体",
          m.subject_alpha(rgb).getextrema() == (255, 255))
    rgba = rgb.convert("RGBA")
    rgba.putalpha(Image.new("L", rgba.size, 77))
    check("RGBA ⇒ 取 alpha 通道",
          m.subject_alpha(rgba).getextrema() == (77, 77))
    check("mask 优先于 alpha",
          m.subject_alpha(rgba, mask).getextrema() == (0, 255))

    # ---------- N4 几何：纯平移（squash=1, blur=0） ----------
    alpha = mask
    a0 = centroid(alpha)
    sh_rgba, sh_a, off, peak = m.build_shadow(alpha, distance=40, angle=90,
                                             blur=0, opacity=1.0, squash=1.0,
                                             fade=0.0)
    a1 = centroid(sh_a)
    check("纯平移：质心偏移 == (0,40)",
          abs(a1[0] - a0[0]) < 1.01 and abs(a1[1] - a0[1] - 40) < 1.01,
          f"偏移=({a1[0]-a0[0]:.1f},{a1[1]-a0[1]:.1f})")
    check("纯平移：投影框 == 原框 + (0,40)",
          m.content_bbox(sh_a) == (100, 140, 300, 440),
          f"实得 {m.content_bbox(sh_a)}")

    # ---------- N4 压扁：锚在落地线（主体包围盒底边） ----------
    _, sh_a2, _, _ = m.build_shadow(alpha, distance=0, angle=90, blur=0,
                                    opacity=1.0, squash=0.5, fade=0.0)
    bb2 = m.content_bbox(sh_a2)
    check("压扁 0.5：高度 ≈ 原高 × 0.5", bb2[3] - bb2[1] == round(300 * 0.5),
          f"影高={bb2[3]-bb2[1]}")
    check("压扁：底边仍落在落地线（y=399）", bb2[3] == 399 + 1, f"底边={bb2[3]}")

    # ---------- N4 blur 单调：覆盖像素数随 blur 递增 ----------
    counts = []
    for b in (0, 8, 24):
        _, sh_b, _, _ = m.build_shadow(alpha, distance=0, angle=90, blur=b,
                                       opacity=1.0, squash=1.0, fade=0.0)
        counts.append(nonzero(sh_b))
    check("blur 0<8<24 ⇒ 覆盖像素数单调增", counts[0] < counts[1] < counts[2],
          f"{counts}")

    # ---------- N5 透明度：峰值 alpha ≈ opacity×255 ----------
    for op, want in ((0.45, 115), (1.0, 255), (0.2, 51)):
        _, _, _, pk = m.build_shadow(alpha, distance=0, angle=90, blur=0,
                                     opacity=op, squash=1.0, fade=0.0)
        check(f"峰值 alpha == round({op}×255)", pk == want, f"实得 {pk}（期望 {want}）")

    # ---------- fade：远端（下侧）更淡 ----------
    _, sh_nf, _, _ = m.build_shadow(alpha, distance=40, angle=90, blur=0,
                                    opacity=1.0, squash=1.0, fade=0.0)
    _, sh_fd, _, _ = m.build_shadow(alpha, distance=40, angle=90, blur=0,
                                    opacity=1.0, squash=1.0, fade=0.9)
    def band_mean(img, y0, y1):
        px = img.crop((0, y0, img.width, y1)).load()
        s = n = 0
        for y in range(y1 - y0):
            for x in range(img.width):
                v = px[x, y]
                if v:
                    s += v
                    n += 1
        return (s / n) if n else 0.0
    # ⚠️ 必须比"均值"：两侧覆盖像素数不同，比总和是假判据
    mt = band_mean(sh_fd, 140, 290)
    mb = band_mean(sh_fd, 290, 440)
    check("fade=0.9：远端（下）均值更淡", mb < mt * 0.6,
          f"上均值={mt:.1f} 下均值={mb:.1f}")
    check("fade=0：上下均值基本相同",
          abs(band_mean(sh_nf, 140, 290) - band_mean(sh_nf, 290, 440)) < 1.0)

    # ---------- N2 确定性 ----------
    args = dict(mask=alpha, distance=30, angle=120, blur=10, opacity=0.4,
                squash=0.6, fade=0.3)
    o1, b1, _ = m.process(Image.new("RGB", (600, 800), (250, 250, 250)), **args)
    o2, b2, _ = m.process(Image.new("RGB", (600, 800), (250, 250, 250)), **args)
    check("同参两跑逐字节一致（确定性）", o1.tobytes() == o2.tobytes() and b1 == b2)

    # ---------- N6 画布模式 ----------
    src = Image.new("RGB", (600, 800), (240, 240, 240))
    img = src.copy()
    img.paste((30, 90, 200), (100, 100, 300, 400))
    o_keep, b_keep, _ = m.process(img, mask=alpha, background=img, distance=0,
                                  angle=90, blur=0, opacity=1.0, squash=1.0,
                                  fade=0.0, canvas="keep")
    check("canvas=keep ⇒ 输出尺寸 == 输入尺寸", o_keep.size == img.size,
          f"{o_keep.size}")
    # 主体靠下 + 位移足够大 ⇒ 投影越出下边界，expand 必须扩画布
    low_mask = Image.new("L", img.size, 0)
    low_mask.paste(Image.new("L", (200, 180), 255), (100, 600))
    o_exp, b_exp, mexp = m.process(img, mask=low_mask, background=img,
                                   distance=80, angle=90, blur=10, opacity=0.5,
                                   squash=0.6, fade=0.0, canvas="expand")
    check("canvas=expand ⇒ 输出更大（容纳投影）", o_exp.size[1] > img.size[1],
          f"{o_exp.size} vs {img.size}")

    # ---------- N3 非投影区逐字节不变 ----------
    outside_identical(img, o_keep, b_keep, "keep+底图")
    # expand：主体区在输出里逐字节保留
    # ⚠️ 只比**主体自己的像素**：影子会落在主体旁边的底图上（那是应该改的），
    # 整块原图矩形比是假判据。主体像素 = low_mask>0 的位置。
    ox, oy = mexp["subject_at"]
    o_px = o_exp.convert("RGB").load()
    i_px = img.load()
    m_px = low_mask.load()
    bad = sum(1 for y in range(img.height) for x in range(img.width)
              if m_px[x, y] > 0 and o_px[x + ox, y + oy] != i_px[x, y])
    check("expand：主体像素逐字节保留（按 subject_at 定位）", bad == 0,
          f"subject_at={mexp['subject_at']} 改动 {bad} px")
    outside_identical(img, o_exp, b_exp, "expand")

    # ---------- RGBA 通路：透明画布不做 alpha 平方 ----------
    half = Image.new("RGBA", (600, 800), (0, 0, 0, 0))
    half.paste((255, 0, 0, 128), (100, 100, 300, 400))
    # 位移把影子推离主体 ⇒ 主体像素处只有"主体自身"，alpha/RGB 必须原样
    o_rgba, _, _ = m.process(half, distance=300, angle=90, blur=0, opacity=0.5,
                             squash=1.0, fade=0.0, canvas="keep")
    a_out = o_rgba.getchannel("A").getpixel((200, 250))
    rgb_out = o_rgba.convert("RGB").getpixel((200, 250))
    check("RGBA 通路：主体半透明像素 alpha 不被平方", a_out == 128, f"实得 {a_out}")
    check("RGBA 通路：主体 RGB 不被影子污染", rgb_out == (255, 0, 0), f"实得 {rgb_out}")

    # ---------- expand + background：画布被撑大后不许留白条（exec 716 实测白条） ----------
    bg_strip = Image.new("RGB", (200, 300), (10, 200, 30))
    # 左右列与上下行给不同颜色，便于断言"延展用的是边缘像素"
    for yy in range(300):
        bg_strip.putpixel((0, yy), (11, 22, 33))
        bg_strip.putpixel((199, yy), (44, 55, 66))
    for xx in range(200):
        bg_strip.putpixel((xx, 299), (77, 88, 99))
    small = Image.new("RGBA", (120, 100), (0, 0, 0, 0))
    small.paste((200, 30, 30, 255), (10, 10, 90, 80))
    o_band, b_band, m_band = m.process(small, background=bg_strip, distance=260,
                                       angle=90, blur=6, opacity=0.5, squash=0.5,
                                       canvas="expand")
    white_like = sum(1 for y in range(o_band.height) for x in range(o_band.width)
                     if o_band.getpixel((x, y)) == (255, 255, 255))
    check("expand+background：画布无纯白补边", white_like == 0,
          f"白像素 {white_like}px，画布 {o_band.size}")
    # 底部延展条应与背景最后一行（77,88,99）同色（除被影子压暗的部分之外）
    y_last = o_band.height - 1
    edge = o_band.getpixel((o_band.width - 3, y_last))
    check("expand+background：底部补边取背景边缘像素",
          abs(edge[0] - 77) <= 2 and abs(edge[1] - 88) <= 2 and abs(edge[2] - 99) <= 2,
          f"实得 {edge}（期望≈(77,88,99)）")

    # ---------- 空蒙版 / 无主体 ----------
    empty = Image.new("L", (100, 100), 0)
    out_e, box_e, _ = m.process(Image.new("RGB", (100, 100), (255, 255, 255)),
                                mask=empty, distance=20, angle=90, blur=4)
    check("空蒙版 ⇒ 无投影框、输出等于输入", box_e is None and
          out_e.convert("RGB").tobytes() ==
          Image.new("RGB", (100, 100), (255, 255, 255)).tobytes())

    smoke_main()
    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


def smoke_main():
    """main() 桩冒烟：桩掉 flowx_client/image_io，真跑一遍 main()。

    为什么必须有：`node mock` 跑的是 mock.py，跑不到 main.py（image-watermark 就因此
    把 `NameError: opacity` 带到了真机）。这里任何未定义名字/参数名写错都会本机暴露。
    """
    import sys as _sys
    import types as _types
    calls = {}

    def _param(key, default=None, cast=None):
        vals = {"service_url": "http://stub", "image": "obj-1", "mask": "",
                "background": "", "distance": "32", "angle": "90", "blur": "12",
                "opacity": "0.5", "squash": "0.6", "fade": "0.2",
                "color": "#203040", "canvas": "expand", "service_url_host": "",
                "service_token": ""}
        v = vals.get(key, default)
        return cast(v) if cast else v

    stub = _types.ModuleType("flowx_client")
    stub.param = _param
    stub.token = lambda: "tk"
    stub.emit = lambda **kw: calls.update({"emit": kw})
    stub.emit_result_preview = lambda *a, **k: calls.update({"preview": a})
    _sys.modules["flowx_client"] = stub

    io_stub = _types.ModuleType("image_io")
    io_stub.get_pimg = lambda base, oid, tok=None, mode=None: (
        Image.new("L", (200, 300), 255) if str(mode).upper() == "L"
        else Image.new("RGBA", (200, 300), (10, 200, 30, 255)))
    io_stub.post_pimg = lambda base, pil, tok=None: calls.update(
        {"out_size": pil.size}) or "obj-out"
    _sys.modules["image_io"] = io_stub

    import io as _io
    from contextlib import redirect_stdout
    buf = _io.StringIO()
    try:
        with redirect_stdout(buf):
            smoke_main_mod = load_mod()
        with redirect_stdout(buf):
            smoke_main_mod.main()
        check("main() 桩冒烟跑通", "emit" in calls and "out_size" in calls,
              f"emit={calls.get('emit')}")
        box = calls.get("emit", {}).get("shadow_box", "")
        check("main() 输出 shadow_box", len(box.split(",")) == 4, f"box={box!r}")
        # ⚠️ 平台陷阱（exec 714 实测）：Studio 的 codec-block 提取器用 `\s*#.*$`
        # 无差别删行尾注释，引号里的 # 也会被截断 ⇒ 整块 YAML 解析失败、输出全丢。
        # 因此 emit 出去的任何字符串都不许带 #（也不许带 ```）。
        bad = [f"{k}={v!r}" for k, v in calls.get("emit", {}).items()
               if isinstance(v, str) and ("#" in v or "```" in v)]
        check("emit 值不含 # 与 ``` （codec-block 提取器陷阱）", not bad, str(bad))
    except Exception as e:  # noqa: BLE001
        check("main() 桩冒烟跑通", False, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
