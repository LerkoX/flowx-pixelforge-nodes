#!/usr/bin/env python3
"""matting-composite 本机 PIL 逻辑单测（不依赖推理服务/GPU/numpy）。

用法：python3 nodes/matting-composite/test_main.py
"""
import importlib.util
import pathlib
import sys

from PIL import Image

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("m", HERE / "main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FAILED = 0


def check(label, cond):
    global FAILED
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        FAILED += 1


def rgba_split(w, h):
    """左半：不透明红；右半：全透明（模拟抠好的 RGBA 前景）。"""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for x in range(w // 2):
        for y in range(h):
            im.putpixel((x, y), (255, 0, 0, 255))
    return im


W = H = 32
fg = rgba_split(W, H)

out = m.process(fg, bg_color="#0000FF")
check("尺寸 = 前景尺寸", out.size == (W, H))
check("模式 = RGB（落地/落盘用）", out.mode == "RGB")
check("全透明区 == 纯底色（逐像素）", out.getpixel((W - 1, 0)) == (0, 0, 255))
check("不透明区 == 前景色", out.getpixel((0, 0)) == (255, 0, 0))

# mask 覆盖 alpha
mask = Image.new("L", (W, H), 0)
for x in range(W // 2, W):
    for y in range(H):
        mask.putpixel((x, y), 255)
out_mask = m.process(fg.convert("RGB"), mask=mask, bg_color="#00FF00")
check("mask 覆盖后右半（mask=255）显示前景黑色而非绿底",
      out_mask.getpixel((W - 1, 0)) == (0, 0, 0))
check("mask 覆盖后左半（mask=0）显示绿底", out_mask.getpixel((0, 0)) == (0, 255, 0))

# 羽化：半透过渡 → 非纯色混合
edge = Image.new("RGBA", (W, H), (0, 0, 0, 0))
for x in range(W):
    for y in range(H):
        edge.putpixel((x, y), (255, 255, 255, 255 if x < W // 2 else 0))
soft = m.process(edge, bg_color="#000000", feather=3)
mid = soft.getpixel((W // 2, H // 2))
check("feather=3 在边界产生中间灰度（非纯黑/纯白）", 0 < mid[0] < 255)

# 偏移
shift = m.process(fg, bg_color="#000000", x=4)
check("x=+4 前景右移 4px（x=4 处为前景红）", shift.getpixel((4, 0)) == (255, 0, 0))
check("x=+4 后 x=3 仍为底色", shift.getpixel((3, 0)) == (0, 0, 0))

# bg_mode=image：背景图按 cover 铺满（更小的底图被放大到前景尺寸，不产生黑边）
bg = Image.new("RGB", (8, 8), (0, 255, 0))
out_img = m.process(fg, bg_mode="image", bg_img=bg)
check("bg_mode=image → 透明区为背景图绿色", out_img.getpixel((W - 1, 0)) == (0, 255, 0))
check("bg_mode=image → 前景区仍为前景色", out_img.getpixel((0, 0)) == (255, 0, 0))

try:
    m.process(fg, bg_mode="image")
    check("bg_mode=image 且缺 bg_image → 报 ValueError", False)
except ValueError as e:
    check("bg_mode=image 且缺 bg_image → 报 ValueError", "bg_image" in str(e))

# 颜色解析
check("#FFF 与 #FFFFFF 等价", m._parse_color("#FFF") == (255, 255, 255, 255))
check("#RRGGBBAA 支持", m._parse_color("#11223344") == (17, 34, 51, 68))
try:
    m._parse_color("#12345")
    check("非法色值 → ValueError", False)
except ValueError:
    check("非法色值 → ValueError", True)

print()
print("FAILED =", FAILED)
sys.exit(1 if FAILED else 0)
