#!/usr/bin/env python3
"""第一档 9 个图像/蒙版节点的本机 PIL 逻辑单测（不依赖推理服务）。

用法：python3 nodes/test-tier1-nodes.py
逐节点把节点目录加入 sys.path 后按文件位置导入 main.py，直接测其 process()。
"""
import importlib.util
import pathlib
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).parent


def load(name):
    d = ROOT / name
    sys.path.insert(0, str(d))
    try:
        spec = importlib.util.spec_from_file_location(f"{name}.main", d / "main.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.pop(0)


def solid(w, h, rgb):
    return Image.new("RGB", (w, h), rgb)


def gray(w, h, v):
    return Image.new("L", (w, h), v)


failed = 0


def check(label, cond):
    global failed
    print(("PASS" if cond else "FAIL"), label)
    if not cond:
        failed += 1


# --- image-rotate ---
m = load("image-rotate")
out = m.process(solid(100, 50, (255, 0, 0)), 90, expand=True)
check("rotate 90 expand -> 50x100", out.size == (50, 100))
out = m.process(solid(100, 50, (255, 0, 0)), 90, expand=False)
check("rotate 90 no-expand keeps 100x50", out.size == (100, 50))
out = m.process(solid(10, 10, (1, 2, 3)), 0)
check("rotate 0 keeps pixels", out.getpixel((5, 5)) == (1, 2, 3))

# --- image-flip ---
m = load("image-flip")
im = solid(2, 1, (255, 0, 0))
im.putpixel((1, 0), (0, 0, 255))
check("flip horizontal", m.process(im, "horizontal").getpixel((0, 0)) == (0, 0, 255))
check("flip vertical 1px-high unchanged", m.process(im, "vertical").getpixel((1, 0)) == (0, 0, 255))

# --- image-crop ---
m = load("image-crop")
im = solid(10, 10, (10, 20, 30))
im.putpixel((3, 4), (200, 100, 50))
out = m.process(im, 3, 4, 2, 2)
check("crop size", out.size == (2, 2))
check("crop origin pixel", out.getpixel((0, 0)) == (200, 100, 50))
for args in [(-1, 0, 2, 2), (0, 0, 0, 2), (8, 0, 5, 2)]:
    try:
        m.process(im, *args)
        check(f"crop invalid {args} rejected", False)
    except RuntimeError:
        check(f"crop invalid {args} rejected", True)

# --- image-composite ---
m = load("image-composite")
bg = solid(10, 10, (0, 0, 255))
fg = solid(4, 4, (255, 0, 0))
out = m.process(bg, fg, None, 2, 3)
check("composite no-mask pixel", out.getpixel((2, 3)) == (255, 0, 0))
check("composite no-mask untouched", out.getpixel((0, 0)) == (0, 0, 255))
check("composite bg not mutated", bg.getpixel((2, 3)) == (0, 0, 255))
mask = gray(4, 4, 0)  # 全黑 = 不贴
out = m.process(bg, fg, mask, 0, 0)
check("composite black mask = no paste", out.getpixel((0, 0)) == (0, 0, 255))
mask = gray(2, 2, 255)  # 尺寸不符 -> 自动缩放到 fg 尺寸，全白 = 全贴
out = m.process(bg, fg, mask, 0, 0)
check("composite mask auto-resize + full paste", out.getpixel((3, 3)) == (255, 0, 0))

# --- mask-from-image ---
m = load("mask-from-image")
im = solid(4, 4, (255, 0, 0))
check("luminance of pure red ≈ 76", abs(m.process(im, "luminance").getpixel((0, 0)) - 76) <= 2)
check("red channel = 255", m.process(im, "red").getpixel((0, 0)) == 255)
check("green channel of red = 0", m.process(im, "green").getpixel((0, 0)) == 0)
try:
    m.process(im, "alpha")
    check("mask-from-image rejects alpha", False)
except RuntimeError:
    check("mask-from-image rejects alpha", True)

# --- mask-to-image ---
m = load("mask-to-image")
out = m.process(gray(4, 4, 128))
check("mask-to-image RGB", out.mode == "RGB" and out.getpixel((0, 0)) == (128, 128, 128))

# --- mask-grow ---
m = load("mask-grow")
mk = gray(9, 9, 0)
for yy in range(3, 6):
    for xx in range(3, 6):
        mk.putpixel((xx, yy), 255)
grown = m.process(mk, 1)
check("grow +1 expands white region", grown.getpixel((2, 2)) == 255)
shrunk = m.process(mk, -1)
check("grow -1 shrinks white region to center", shrunk.getpixel((4, 4)) == 255
      and shrunk.getpixel((3, 3)) == 0 and shrunk.getpixel((4, 3)) == 0)
check("grow 0 identity", m.process(mk, 0).getpixel((4, 4)) == 255)

# --- mask-feather ---
m = load("mask-feather")
mk = gray(11, 11, 0)
for yy in range(5, 11):
    for xx in range(11):
        mk.putpixel((xx, yy), 255)
out = m.process(mk, 2.0)
edge = out.getpixel((5, 4))
check("feather softens edge (0<px<255)", 0 < edge < 255)
check("feather radius 0 identity", m.process(mk, 0).getpixel((0, 6)) == 255)

# --- mask-invert ---
m = load("mask-invert")
out = m.process(gray(2, 2, 30))
check("invert 30 -> 225", out.getpixel((0, 0)) == 225)

print()
if failed:
    print(f"{failed} FAILED")
    sys.exit(1)
print("ALL PASS")
