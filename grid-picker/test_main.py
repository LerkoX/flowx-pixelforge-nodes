#!/usr/bin/env python3
"""grid-picker 本机 PIL 逻辑单测（不依赖推理服务/GPU/numpy）。

用法：python3 grid-picker/test_main.py
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


# 1) 焦点裁方：竖图 fy=0 取顶部，fy=1 取底部
v = Image.new("RGB", (100, 300))  # 竖图：上红中绿下蓝
for y in range(300):
    for x in range(100):
        v.putpixel((x, y), (255, 0, 0) if y < 100 else ((0, 255, 0) if y < 200 else (0, 0, 255)))
top = m.cover_crop_focus(v, 50, 0.5, 0.0)
mid = m.cover_crop_focus(v, 50, 0.5, 0.5)
bot = m.cover_crop_focus(v, 50, 0.5, 1.0)
check("fy=0 取顶部红", top.getpixel((25, 25)) == (255, 0, 0))
check("fy=0.5 取中部绿", mid.getpixel((25, 25)) == (0, 255, 0))
check("fy=1 取底部蓝", bot.getpixel((25, 25)) == (0, 0, 255))
check("输出尺寸=side", top.size == (50, 50))

# 2) 横图 fx 同理
h = Image.new("RGB", (300, 100))
for y in range(100):
    for x in range(300):
        h.putpixel((x, y), (255, 0, 0) if x < 100 else ((0, 255, 0) if x < 200 else (0, 0, 255)))
left = m.cover_crop_focus(h, 50, 0.0, 0.5)
right = m.cover_crop_focus(h, 50, 1.0, 0.5)
check("fx=0 取左部红", left.getpixel((25, 25)) == (255, 0, 0))
check("fx=1 取右部蓝", right.getpixel((25, 25)) == (0, 0, 255))

# 3) 方图直接缩放
sq = m.cover_crop_focus(Image.new("RGB", (200, 200), (1, 2, 3)), 100, 0.2, 0.8)
check("方图直通", sq.size == (100, 100) and sq.getpixel((50, 50)) == (1, 2, 3))

# 4) crops 解析：缺省 0.5、越界钳位、坏值回退
c = m.parse_crops('[{"fx":0.2,"fy":0.8},{"fx":2,"fy":-1},{}]', 4)
check("crops 解析", c[0] == (0.2, 0.8))
check("crops 越界钳位", c[1] == (1.0, 0.0))
check("crops 坏值回退 0.5", c[2] == (0.5, 0.5) and c[3] == (0.5, 0.5))
check("crops 空串全默认", m.parse_crops("", 2) == [(0.5, 0.5), (0.5, 0.5)])

# 5) files 解析
import os
import tempfile
with tempfile.TemporaryDirectory() as td:
    p = pathlib.Path(td, "a.jpg")
    p.write_bytes(b"x")
    fs = m.parse_files(__import__('json').dumps([str(p), "/nonexistent/b.jpg"]))
    check("files 过滤不存在", fs == [str(p)])
    check("files 空给 []", m.parse_files("") == [])

# 6) compose 版式（与 grid-compose 一致：top_banner 撑起白区）
cells = [Image.new("RGB", (100, 100), (i * 20, 0, 0)) for i in range(9)]
out = m.compose(cells, cell=100, gap=10, margin=10, top_banner=100)
check("compose 尺寸 340x440", out.size == (340, 440))
check("banner 区背景色", out.getpixel((170, 30)) == (255, 255, 255))
try:
    m.compose(cells[:8])
    check("8 张应报错", False)
except ValueError:
    check("8 张报 ValueError", True)

print()
if FAILED:
    print(f"{FAILED} 项失败")
    sys.exit(1)
print("全部通过")
