#!/usr/bin/env python3
"""image-channel-merge 本机 PIL 逻辑单测（不依赖推理服务）。

用法：python3 nodes/image-channel-merge/test_main.py
"""
import importlib.util
import pathlib
import random
import sys

from PIL import Image

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))


def load(mod, path):
    spec = importlib.util.spec_from_file_location(mod, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


m = load("merge_main", HERE / "main.py")
filters = load("image_filters", HERE / "image_filters.py")

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

r = filters.channel_extract(im, "red")
g = filters.channel_extract(im, "green")
b = filters.channel_extract(im, "blue")

out = m.process(r, g, b)
check("三路合并：尺寸/模式保持", out.size == im.size and out.mode == "RGB")
check("三路合并 == 原图（拆分再合并逐像素一致）", out.tobytes() == im.convert("RGB").tobytes())

out_a = m.process(r, g, b, filters.channel_extract(im, "alpha"))
check("给 alpha → RGBA 且 alpha 全 255",
      out_a.mode == "RGBA" and out_a.split()[3].getextrema() == (255, 255))

out_s = m.process(r, g, filters.channel_extract(im, "blue").resize((24, 16)))
check("尺寸不符自动缩放对齐到 R 路尺寸", out_s.size == im.size)

out_nonl = m.process(im, im, im)
check("非灰度输入自动转 L", out_nonl.tobytes() == im.convert("L").convert("RGB").tobytes())

try:
    m.process(r, g, None)
    check("缺通道抛 ValueError", False)
except ValueError:
    check("缺通道抛 ValueError", True)

print()
print("FAILED =", FAILED)
sys.exit(1 if FAILED else 0)
