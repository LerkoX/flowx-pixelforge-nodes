#!/usr/bin/env python3
"""grid-compose 本机 PIL 逻辑单测（不依赖推理服务/GPU/numpy）。

用法：python3 grid-compose/test_main.py
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


def solid(w, h, rgb):
    return Image.new("RGB", (w, h), rgb)


cells = [solid(800, 600, (i * 25, 0, 0)) for i in range(9)]

# 1) 尺寸公式：size = margin*2 + cell*3 + gap*2
out = m.compose(cells, cell=300, gap=10, margin=20)
check("默认排版尺寸 300*3+10*2+20*2=960", out.size == (960, 960))
check("输出 RGB 无 alpha", out.mode == "RGB")

# 2) 间隙露出背景色（默认白），格心是各自颜色
px = out.convert("RGB")
cx = 20 + 150  # 第一格中心 x
cy = 20 + 150
check("格心是 cell 图颜色", px.getpixel((cx, cy)) == (0, 0, 0))
gapx = 20 + 300 + 5  # 第一、二格之间的间隙
check("间隙是背景色(白)", px.getpixel((gapx, cy)) == (255, 255, 255))
check("外边距是背景色(白)", px.getpixel((5, cy)) == (255, 255, 255))

# 3) cover 裁切：竖图裁两侧、横图裁上下，不畸变
wide = solid(1200, 300, (0, 255, 0))
out2 = m.compose([wide] * 9, cell=100, gap=0, margin=0, bg="#000000")
check("横图 cover 后格心仍是绿色", out2.getpixel((50, 50)) == (0, 255, 0))

# 4) 圆角：radius>0 时格子角落露出背景色
out3 = m.compose(cells, cell=200, gap=0, margin=0, radius=40, bg="#FFFFFF")
check("圆角角落露出背景色", out3.getpixel((2, 2)) == (255, 255, 255))
check("圆角格心不变", out3.getpixel((100, 100)) == (0, 0, 0))

# 5) 张数校验
try:
    m.compose(cells[:8])
    check("8 张应报错", False)
except ValueError as e:
    check("8 张报 ValueError", "9" in str(e))

# 6) 背景色解析
check("#FFF 展开", m._parse_color("#FFF") == (255, 255, 255))
try:
    m._parse_color("#ZZZZZZ")
    check("非法色值应报错", False)
except ValueError:
    check("非法色值报 ValueError", True)

# 7) top_banner：顶部加白区，格子整体下移，画布变矩形
out4 = m.compose(cells, cell=100, gap=10, margin=10, top_banner=100)
check("banner 画布 340x440", out4.size == (340, 440))
check("banner 区是背景色", out4.getpixel((170, 30)) == (255, 255, 255))
check("第一格下移到 banner 之下", out4.getpixel((60, 100 + 10 + 50)) == (0, 0, 0))

print()
if FAILED:
    print(f"{FAILED} 项失败")
    sys.exit(1)
print("全部通过")
