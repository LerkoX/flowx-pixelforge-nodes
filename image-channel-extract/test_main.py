#!/usr/bin/env python3
"""image-channel-extract 本机 PIL 逻辑单测（不依赖推理服务）。

用法：python3 nodes/image-channel-extract/test_main.py
"""
import importlib.util
import pathlib
import random
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


rng = random.Random(20261010)
im = Image.new("RGB", (48, 32))
im.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(48 * 32)])

lum = m.process(im, channel="luminance")
check("luminance：单通道 L 且尺寸保持", lum.mode == "L" and lum.size == im.size)
check("luminance == 原图 L 转换", lum.tobytes() == im.convert("L").tobytes())

red = m.process(im, channel="red")
check("red == 原图 R 通道", red.tobytes() == im.split()[0].tobytes())
green = m.process(im, channel="green")
check("green == 原图 G 通道", green.tobytes() == im.split()[1].tobytes())
blue = m.process(im, channel="blue")
check("blue == 原图 B 通道", blue.tobytes() == im.split()[2].tobytes())

check("无 alpha 原图 → alpha 通道全白 255",
      m.process(im, channel="alpha").getextrema() == (255, 255))
rgba = im.convert("RGBA")
rgba.putalpha(Image.new("L", im.size, 77))
check("带 alpha 原图 → 取真实 alpha",
      m.process(rgba, channel="alpha").getextrema() == (77, 77))

try:
    m.process(im, channel="zzz")
    check("非法通道抛 ValueError", False)
except ValueError:
    check("非法通道抛 ValueError", True)

try:
    m.process(im, channel=None)
    check("channel=None 回退 luminance", True)
except Exception as e:  # noqa: BLE001
    check("channel=None 回退 luminance", False)

print()
print("FAILED =", FAILED)
sys.exit(1 if FAILED else 0)
