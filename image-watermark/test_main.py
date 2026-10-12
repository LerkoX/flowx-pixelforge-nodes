#!/usr/bin/env python3
"""image-watermark 本机单测（纯 PIL，零 numpy；不需要 GPU/推理服务）。

重点验证「水印只改水印区域」这条核心契约（电商交付要能证明原图未被污染）。
用法：python3 nodes/image-watermark/test_main.py
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
    spec = importlib.util.spec_from_file_location("wm_main", HERE / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE = Image.new("RGB", (200, 100), (30, 60, 120))
for y in range(100):
    for x in range(200):
        if (x // 10 + y // 10) % 2:
            BASE.putpixel((x, y), (180, 180, 180))


def outside_identical(orig, new, box, label):
    """水印区之外必须逐字节不变。"""
    ox, oy = orig.load(), new.load()
    x0, y0, w, h = box
    diff = 0
    for y in range(orig.height):
        for x in range(orig.width):
            if x0 <= x < x0 + w and y0 <= y < y0 + h:
                continue
            if ox[x, y] != oy[x, y]:
                diff += 1
    check(f"{label} 水印区外逐字节不变", diff == 0, f"区外改动 {diff} px")


def inside_changed(orig, new, label):
    check(f"{label} 水印区内确有改动", orig.tobytes() != new.tobytes())


def main():
    m = load_mod()

    check("颜色解析 #RRGGBB", m._parse_color("#FFFFFF") == (255, 255, 255, 255))
    check("颜色解析 #RRGGBBAA", m._parse_color("#FF000080") == (255, 0, 0, 128))
    check("颜色解析 r,g,b", m._parse_color("10,20,30") == (10, 20, 30, 255))
    check("颜色解析空值用默认", m._parse_color("", (1, 2, 3)) == (1, 2, 3, 255))
    for bad in ("#FFF", "1,2", "300,0,0", "zz"):
        try:
            m._parse_color(bad)
            check(f"非法颜色必须报错 {bad!r}", False, "未报错")
        except ValueError:
            check(f"非法颜色报错 {bad!r}", True)

    pos_expect = {
        "top-left": (24, 24), "top-center": (None, 24), "top-right": (None, 24),
        "middle-left": (24, None), "center": (None, None), "middle-right": (None, None),
        "bottom-left": (24, None), "bottom-center": (None, None),
        "bottom-right": (None, None),
    }
    for p in m.POSITIONS:
        x, y = m._pos_xy(p, (200, 100), (40, 20), 24)
        ok = (0 <= x <= 200 - 40) and (0 <= y <= 100 - 20)
        check(f"九宫格 {p} 坐标在界内", ok, f"({x},{y})")
    check("top-left 精确=边距", m._pos_xy("top-left", (200, 100), (40, 20), 24) == (24, 24))
    check("bottom-right 精确=右下-边距",
          m._pos_xy("bottom-right", (200, 100), (40, 20), 24) == (200 - 40 - 24, 100 - 20 - 24))
    check("center 居中", m._pos_xy("center", (200, 100), (40, 20), 24) == (80, 40))

    out, box, item = m.process(BASE, text="SAMPLE 2026", position="bottom-right",
                               margin=10, opacity=1.0, font_size=18)
    check("尺寸不变", out.size == BASE.size, str(out.size))
    check("box 为 4 元组且在水印区内", len(box) == 4 and box[0] >= 0 and box[1] >= 0,
          str(box))
    outside_identical(BASE, out, box, "文字水印")
    inside_changed(BASE, out, "文字水印")

    _o2, box2, _ = m.process(BASE, text="X", position="top-left", margin=4,
                             opacity=0.3, font_size=14)
    check("不同定位 box 不同", box2 != box, f"{box2} vs {box}")
    _o3, box3, _ = m.process(BASE, text="X", position="center", margin=0,
                             opacity=1.0, font_size=40)
    check("center 定位 box 居中", abs((box3[0] + box3[2] // 2) - 100) <= 2, str(box3))

    _ot, boxt, itemt = m.process(BASE, text="T", opacity=0.25, font_size=12,
                                 tile=True, angle=30, tile_gap=20)
    check("平铺时 box = 整图", boxt == (0, 0, 200, 100), str(boxt))
    check("平铺改变了画面", BASE.tobytes() != _ot.tobytes())

    logo = Image.new("RGBA", (50, 20), (255, 0, 0, 255))
    _ol, boxl, iteml = m.process(BASE, logo=logo, position="top-left", margin=6,
                                 opacity=0.8, logo_scale=0.5)
    check("Logo 按 scale 缩放", iteml == (25, 10), str(iteml))
    check("Logo 模式下 box 与图元同尺寸", (boxl[2], boxl[3]) == iteml, str(boxl))
    outside_identical(BASE, _ol, boxl, "Logo 水印")

    try:
        m.process(BASE, text="", logo=None)
        check("text/logo 全空必须报错", False, "未报错")
    except ValueError:
        check("text/logo 全空报错", True)
    # 空值按设计回退到默认 bottom-right（宽松语义），非法**值**才报错
    _od, boxd, _ = m.process(BASE, text="x", position="")
    _obr, boxbr, _ = m.process(BASE, text="x", position="bottom-right")
    check("position 空值回退到默认 bottom-right", boxd == boxbr, f"{boxd} vs {boxbr}")
    for bad in ("middle", "diagonal", "top-middle"):
        try:
            m.process(BASE, text="x", position=bad)
            check(f"非法 position 必须报错 {bad!r}", False, "未报错")
        except ValueError:
            check(f"非法 position 报错 {bad!r}", True)
    for bad in (0.0, 1.5, -0.2):
        try:
            m.process(BASE, text="x", opacity=bad)
            check(f"非法 opacity 必须报错 {bad!r}", False, "未报错")
        except ValueError:
            check(f"非法 opacity 报错 {bad!r}", True)

    smoke_main()

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


def smoke_main():
    """main() 桩冒烟：桩掉 flowx_client 的网络/emit，真跑一遍 main()。

    为什么必须有：`node mock` 跑的是 mock.py，**跑不到 main.py** —— image-watermark
    就因此把一个 `NameError: opacity` 带到了真机（B4/S2 首次 e2e 失败）。这里用桩把
    main() 完整跑一次，任何未定义名字/参数名写错/漏传都会在本机暴露。
    """
    import sys as _sys
    import types as _types
    calls = {}

    def _param(key, default=None, cast=None):
        vals = {"service_url": "http://stub", "image": "obj-1", "logo": "",
                "text": "SAMPLE", "position": "bottom-left", "margin": "7",
                "opacity": "0.5", "font_size": "18", "font_path": "",
                "color": "#FF0000", "stroke_width": "1",
                "stroke_color": "#000000", "logo_scale": "0.0", "tile": "false",
                "angle": "0", "tile_gap": "0", "service_url_host": ""}
        v = vals.get(key, default)
        return cast(v) if cast else v

    stub = _types.ModuleType("flowx_client")
    stub.param = _param
    stub.token = lambda: "tk"
    stub.emit = lambda **kw: calls.update({"emit": kw})
    stub.emit_result_preview = lambda *a, **k: calls.update({"preview": a})
    stub.get_bytes = lambda *a, **k: b""
    stub.post_bytes = lambda *a, **k: {"id": "obj-out"}
    _sys.modules["flowx_client"] = stub

    io_stub = _types.ModuleType("image_io")
    io_stub.get_pimg = lambda base, oid, tok=None, mode=None: Image.new(
        "RGB", (120, 80), (10, 20, 30))
    io_stub.post_pimg = lambda base, pil, tok=None: "obj-out"
    io_stub.pbool = lambda raw: str(raw).strip().lower() in ("1", "true", "yes", "on")
    io_stub.RESAMPLE = {}
    _sys.modules["image_io"] = io_stub

    m = load_mod()
    try:
        m.main()
        check("main() 桩冒烟无异常（名字未定义/参数写错都会在此暴露）", True)
        check("main() 上报了 image 与 box",
              calls.get("emit", {}).get("image") == "obj-out"
              and "box" in calls.get("emit", {}),
              str({k: v for k, v in calls.get("emit", {}).items() if k != "image"}))
        check("main() 上报了预览帧", "preview" in calls)
    except Exception as e:
        check("main() 桩冒烟无异常", False, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    sys.exit(main())
