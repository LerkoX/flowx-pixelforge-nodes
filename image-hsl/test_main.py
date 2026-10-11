#!/usr/bin/env python3
"""image-hsl 本机 PIL 逻辑单测（不依赖推理服务）。

用法：python3 nodes/image-hsl/test_main.py
直接导入同目录 main.py（其依赖 flowx_client / image_io / image_filters 均在同目录）。
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


def px(im):
    return im.convert("RGB").tobytes()


def sd(im):
    return ImageStat.Stat(im.convert("L")).stddev[0]


def mean(im):
    return ImageStat.Stat(im.convert("L")).mean[0]


rng = random.Random(20261010)
im = Image.new("RGB", (48, 32))
im.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256))
            for _ in range(48 * 32)])
out = m.process(im, hue=0.0, saturation=1.0, lightness=0.0)
check("默认参数幂等（逐像素一致）", px(out) == px(im))
check("尺寸与模式保持", out.size == im.size and out.mode == "RGB")
eff = m.process(im, saturation=0.0)
check("saturation=0 -> 三通道相等（灰度）", eff.split()[0].tobytes() == eff.split()[2].tobytes())

print()
print("FAILED =", FAILED)
sys.exit(1 if FAILED else 0)
