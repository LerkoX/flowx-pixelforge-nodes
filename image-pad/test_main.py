#!/usr/bin/env python3
"""image-pad 本机 PIL 逻辑单测（不依赖推理服务）。

用法：python3 nodes/image-pad/test_main.py
"""
import importlib.util
import pathlib
import random
import sys

from PIL import Image, ImageStat

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


def region_bytes(im, box):
    return im.crop(box).tobytes()


rng = random.Random(20261010)
im = Image.new("RGB", (40, 24))
im.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(40 * 24)])

# ---- 扩图画布 ----
out, mask = m.process(im, left=8, right=16, top=4, bottom=12, feather=0)
check("扩边尺寸正确（40+8+16, 24+4+12）", out.size == (64, 40))
check("原图区域逐像素保留", region_bytes(out, (8, 4, 48, 28)) == im.tobytes())
check("蒙版尺寸与扩边图一致", mask.size == out.size and mask.mode == "L")
check("蒙版原图区 0 / 新扩区 255",
      mask.getpixel((20, 20)) == 0 and mask.getpixel((0, 0)) == 255
      and mask.getpixel((63, 39)) == 255)

# edge 填充：边缘像素拉伸（左边界整列等于原图第 0 列）
check("edge 左侧补边 = 原图第 0 列拉伸", region_bytes(out, (0, 4, 8, 28))
      == im.crop((0, 0, 1, 24)).resize((8, 24), Image.Resampling.NEAREST).tobytes())
check("edge 上侧补边 = 原图第 0 行拉伸", region_bytes(out, (8, 0, 48, 4))
      == im.crop((0, 0, 40, 1)).resize((40, 4), Image.Resampling.NEAREST).tobytes())

# ---- 纯色填充 ----
for name, color in (("black", (0, 0, 0)), ("white", (255, 255, 255)), ("gray", (128, 128, 128))):
    o, _ = m.process(im, left=6, right=0, top=0, bottom=0, feather=0, fill=name)
    check(f"fill={name} 新扩区为纯色", o.getpixel((2, 10)) == color and o.size == (46, 24))

# ---- 羽化 ----
_, hard = m.process(im, right=32, feather=0)
_, soft = m.process(im, right=32, feather=16)
band = [soft.getpixel((x, 12)) for x in range(30, 50)]
check("feather>0 时边界出现中间灰度（过渡带）",
      any(0 < v < 255 for v in band)
      and soft.getpixel((40, 12)) < 255
      and hard.getpixel((40, 12)) == 255)

# ---- 零扩边（幂等）----
o0, m0 = m.process(im, 0, 0, 0, 0)
check("零扩边：图像尺寸不变且逐像素一致", o0.size == im.size and o0.tobytes() == im.convert("RGB").tobytes())
check("零扩边：蒙版全 0", m0.getextrema() == (0, 0))

# ---- 参数校验 ----
try:
    m.process(im, left=-1)
    check("负扩边抛 ValueError", False)
except ValueError:
    check("负扩边抛 ValueError", True)
try:
    m.process(im, right=8, fill="zzz")
    check("非法 fill 抛 ValueError", False)
except ValueError:
    check("非法 fill 抛 ValueError", True)

# 只扩单边时其余边尺寸不动
o_top, _ = m.process(im, top=10)
check("仅上扩边：宽度不变、高度 +10", o_top.size == (40, 34))

print()
print("FAILED =", FAILED)
sys.exit(1 if FAILED else 0)
