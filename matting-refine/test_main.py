#!/usr/bin/env python3
"""matting-refine 本机 PIL 逻辑单测（不依赖推理服务/GPU/numpy）。

用法：python3 nodes/matting-refine/test_main.py
"""
import importlib.util
import pathlib
import sys

from PIL import Image, ImageDraw

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


def area(im, thr=128):
    """≥thr 的像素数（走直方图，不需要 numpy）。"""
    return sum(im.histogram()[thr:])


def uniq(im):
    return [i for i, c in enumerate(im.histogram()) if c]


def mean(im):
    h = im.histogram()
    return sum(i * c for i, c in enumerate(h)) / float(im.size[0] * im.size[1])


# 底图：左 24px 白（前景 255）+ 8px 灰（128）边界带 + 右侧黑（0）
base = Image.new("L", (64, 32), 0)
d = ImageDraw.Draw(base)
d.rectangle((0, 0, 23, 31), fill=255)
d.rectangle((24, 0, 31, 31), fill=128)

out = m.process(base)
check("默认参数幂等（逐像素一致）", out.tobytes() == base.tobytes())
check("尺寸/模式保持", out.size == base.size and out.mode == "L")

b = m.process(base, threshold=128)
check("threshold=128 后只有 0/255", uniq(b) == [0, 255])
check("threshold=128 前景面积 = 32×32", area(b) == 32 * 32)
check("threshold=129 把灰带划到背景（面积仍 24×32）",
      area(m.process(base, threshold=129)) == 24 * 32)

check("gamma=2.0 前景更饱满（均值上升）", mean(m.process(base, gamma=2.0)) > mean(base))
# 对比度以 127.5 为轴：灰带 128 在 contrast=5 下 → 130（远离轴），黑/白不动
c5 = m.process(base, contrast=5.0)
check("contrast=5.0 把灰带推离 127.5（128→130）", c5.getpixel((28, 16)) == 130)
check("contrast=5.0 不动纯黑/纯白（0/255 保持）",
      c5.getpixel((63, 16)) == 0 and c5.getpixel((0, 16)) == 255)
check("gamma + contrast 组合不越界（仍 0~255）",
      uniq(m.process(base, gamma=3.0, contrast=3.0))[0] >= 0)

f = m.process(base, feather=4)
check("feather=4 产生中间灰（有过渡带）", len(uniq(f)) > 3)

g = m.process(base, threshold=128, grow=2)
check("grow=+2 前景面积变大", area(g) > area(b))
s = m.process(base, threshold=128, grow=-2)
check("grow=-2 前景面积变小", area(s) < area(b))

# 去小杂点：大白块 + 2×2 孤立小点（形态学开运算应抹掉小点、保住主体）
blob = Image.new("L", (64, 32), 0)
d2 = ImageDraw.Draw(blob)
d2.rectangle((0, 0, 47, 31), fill=255)
d2.rectangle((60, 2, 61, 3), fill=255)
big = m.process(blob, threshold=128)
small = m.process(blob, threshold=128, despeckle=5)
check("despeckle=5 抹掉 4px 孤立小点", area(big) - area(small) == 4)
check("despeckle=5 主体（48×32）基本保住（面积损失 <5%）",
      area(small) > area(big) * 0.95)

inv = m.process(base, invert=True)
check("invert 反相（255-x）", inv.getpixel((0, 0)) == 0 and inv.getpixel((63, 0)) == 255)

print()
print("FAILED =", FAILED)
sys.exit(1 if FAILED else 0)
