#!/usr/bin/env python3
"""mask-paint 本机 PIL 逻辑单测（不依赖推理服务）。

用法：python3 nodes/mask-paint/test_main.py
直接导入同目录 main.py，测 parse_strokes / rasterize / coverage / overlay_thumb。
"""
import importlib.util
import pathlib
import sys

from PIL import Image

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("mask_paint.main", HERE / "main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FAILED = 0


def check(label, cond):
    global FAILED
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        FAILED += 1


# ---- parse_strokes ----
s, w, h = m.parse_strokes("")
check("空串 → 空笔画 512x512", s == [] and (w, h) == (512, 512))
s, w, h = m.parse_strokes("{bad json")
check("坏 JSON → 空笔画兜底", s == [] and (w, h) == (512, 512))
s, w, h = m.parse_strokes('{"strokes":[{"type":"rect","x":0,"y":0,"w":10,"h":10}],"image_size":[256,128]}')
check("正常解析 + 参照尺寸", len(s) == 1 and (w, h) == (256, 128))
s, w, h = m.parse_strokes('{"strokes":[{"type":"rect"},42,null],"image_size":"x"}')
check("非字典笔画被滤除 + 坏 image_size 兜底", len(s) == 1 and (w, h) == (512, 512))

# ---- rasterize：矩形 ----
mask = m.rasterize([{"type": "rect", "x": 10, "y": 10, "w": 20, "h": 30}], 100, 100)
check("rect 全白区域", mask.crop((10, 10, 30, 40)).getextrema() == (255, 255))
check("rect 外全黑（PIL 坐标闭区间）", mask.crop((0, 0, 10, 100)).getextrema() == (0, 0)
      and mask.crop((31, 0, 100, 100)).getextrema() == (0, 0))
check("rect 覆盖率（21x31 闭区间）", abs(m.coverage(mask) - 0.0651) < 0.0001)

# 反向拖拽（负 w/h）归一化
mask = m.rasterize([{"type": "rect", "x": 30, "y": 40, "w": -20, "h": -30}], 100, 100)
check("rect 负宽高归一化", mask.crop((10, 10, 30, 40)).getextrema() == (255, 255))

# ---- rasterize：缩放（参照 50x50 画在 100x100 上）----
mask = m.rasterize([{"type": "rect", "x": 5, "y": 5, "w": 10, "h": 15}], 100, 100, 50, 50)
check("参照系缩放 2x", mask.crop((10, 10, 30, 40)).getextrema() == (255, 255))

# 非等比缩放（512x512 参照 → 768x1152 竖图）
mask = m.rasterize([{"type": "rect", "x": 0, "y": 0, "w": 256, "h": 512}],
                   768, 1152, 512, 512)
check("非等比缩放左半白", mask.crop((0, 0, 384, 1152)).getextrema() == (255, 255)
      and mask.crop((385, 0, 768, 1152)).getextrema() == (0, 0))

# ---- rasterize：笔刷 ----
mask = m.rasterize([{"type": "brush", "size": 20, "points": [[50, 50]]}], 100, 100)
check("单点笔刷=圆斑（直径20）", mask.getpixel((50, 50)) == 255 and mask.getpixel((50, 41)) == 255
      and mask.getpixel((50, 39)) == 0)
mask = m.rasterize([{"type": "brush", "size": 10, "points": [[10, 50], [90, 50]]}], 100, 100)
check("折线笔刷连通", mask.getpixel((50, 50)) == 255 and mask.getpixel((10, 50)) == 255
      and mask.getpixel((90, 50)) == 255)
check("折线笔刷端点圆头", mask.getpixel((7, 50)) == 255)  # 平头帽会漏掉端点半径
check("空 points 笔画跳过", m.rasterize([{"type": "brush", "size": 10, "points": []}],
                                      100, 100).getextrema() == (0, 0))

# ---- erase 后画压先画 ----
mask = m.rasterize([
    {"type": "rect", "x": 0, "y": 0, "w": 100, "h": 100},
    {"type": "erase", "size": 20, "points": [[50, 50]]},
], 100, 100)
check("erase 挖洞", mask.getpixel((50, 50)) == 0 and mask.getpixel((10, 10)) == 255)
mask = m.rasterize([
    {"type": "erase", "size": 20, "points": [[50, 50]]},
    {"type": "rect", "x": 0, "y": 0, "w": 100, "h": 100},
], 100, 100)
check("顺序敏感：erase 先画被覆盖", mask.getpixel((50, 50)) == 255)

# 未知 type 忽略
check("未知笔画类型忽略", m.rasterize([{"type": "lasso", "points": [[1, 1]]}],
                                    100, 100).getextrema() == (0, 0))

# ---- overlay_thumb ----
im = Image.new("RGB", (64, 64), (30, 60, 90))
mask = m.rasterize([{"type": "rect", "x": 0, "y": 0, "w": 32, "h": 64}], 64, 64)
ov = m.overlay_thumb(im, mask)
check("overlay 尺寸一致", ov.size == (64, 64) and ov.mode == "RGB")
check("overlay 白区偏红", ov.getpixel((16, 32))[0] > 150)
check("overlay 黑区近原色", ov.getpixel((48, 32)) == (30, 60, 90))

print()
if FAILED:
    print(f"{FAILED} 项失败")
    sys.exit(1)
print("全部通过")
