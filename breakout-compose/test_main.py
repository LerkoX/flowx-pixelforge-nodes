#!/usr/bin/env python3
"""breakout-compose 本机 PIL 逻辑单测（不依赖推理服务/GPU/numpy）。

用法：python3 breakout-compose/test_main.py
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


def person_rgba(w, h):
    """上半身不透明红、下半身全透明的假人（模拟抠好的 RGBA）。"""
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    for y in range(h // 2):
        for x in range(w):
            im.putpixel((x, y), (255, 0, 0, 255))
    return im


grid = Image.new("RGB", (1000, 1000), (255, 255, 255))

# 1) 落位公式：height_ratio=1.0, feet_y=100 → 人物铺满全图
x0, y0, tw, th = m.person_box((1000, 1000), (500, 1000), 1.0, 50.0, 100.0)
check("身高比 1.0 → th=1000", th == 1000)
check("等比缩放 tw=500", tw == 500)
check("居中 x0=250", x0 == 250)
check("脚底=100% → y0=0", y0 == 0)

# 2) 破框：feet_y=95 + height_ratio=1.0 → y0<0（头顶越出顶边）
x0, y0, tw, th = m.person_box((1000, 1000), (500, 1000), 1.0, 50.0, 95.0)
check("破框 y0<0", y0 == -50)

# 2.5) trim：裁到 alpha 包围盒（半透明人 → 500x500）
trimmed = m.trim_alpha(person_rgba(500, 1000))
check("trim 裁到不透明区", trimmed.size == (500, 500))
try:
    m.trim_alpha(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))
    check("全透明应报错", False)
except ValueError:
    check("全透明报 ValueError", True)

# 3) 合成：人物红色出现在画布内，脚底处
def compose_raw(**kw):
    kw.setdefault("trim", False)  # 合成行为测试用完整图层（保留透明区）
    return m.compose(**kw)

out, box = compose_raw(grid=grid, person=person_rgba(500, 1000), height_ratio=1.0,
                       center_x=50.0, feet_y=100.0, shadow=False)
check("输出尺寸同底图", out.size == (1000, 1000))
check("人物上半身在画面中部可见", out.getpixel((500, 250)) == (255, 0, 0))
check("人物透明区露出底图白", out.getpixel((500, 750)) == (255, 255, 255))

# 4) 破框合成不炸（负 y0 裁剪路径）
out2, box2 = compose_raw(grid=grid, person=person_rgba(500, 1000), height_ratio=1.0,
                         center_x=50.0, feet_y=95.0, shadow=False)
check("破框 person_box y=-50", box2["y"] == -50)
check("破框后人物顶部贴画布顶", out2.getpixel((500, 0)) == (255, 0, 0))

# 5) 投影：开投影后人物右下方出现暗化
out3, _ = compose_raw(grid=grid, person=person_rgba(500, 1000), height_ratio=1.0,
                      center_x=50.0, feet_y=100.0, shadow=True,
                      shadow_distance=60, shadow_angle=90.0, shadow_blur=20,
                      shadow_opacity=0.5)
r, g, b = out3.getpixel((500, 560))  # 人物透明区、正下方偏投影区
check("投影区被暗化（白→灰）", r < 240 and abs(r - g) < 12 and abs(g - b) < 12)

# 6) 关投影 = 纯叠加
out4, _ = compose_raw(grid=grid, person=person_rgba(500, 1000), height_ratio=1.0,
                      center_x=50.0, feet_y=100.0, shadow=False)
check("关投影时透明区纯白", out4.getpixel((500, 560)) == (255, 255, 255))

# 7) 水平翻转
out5, _ = compose_raw(grid=grid, person=person_rgba(500, 1000), height_ratio=1.0,
                      center_x=50.0, feet_y=100.0, flip=True, shadow=False)
check("翻转后仍正常合成", out5.getpixel((500, 250)) == (255, 0, 0))

# 8) trim 默认开：半身人被裁成方形后按真实身高摆放
out6, box6 = m.compose(grid, person_rgba(500, 1000), height_ratio=1.0,
                       center_x=50.0, feet_y=100.0, shadow=False)
check("trim 后身高=底图高", box6["height"] == 1000)
check("trim 后宽度=1000（方形人）", box6["width"] == 1000)

# 9) 透视倾斜：tilt>0 顶角变填充色、顶中仍是网格内容、尺寸不变
red = Image.new("RGB", (1000, 1000), (200, 30, 30))
t = m.tilt_grid(red, 12, "#FFFFFF")
check("tilt 尺寸不变", t.size == (1000, 1000))
check("tilt 左上角变填充色", t.getpixel((2, 2)) == (255, 255, 255))
check("tilt 右上角变填充色", t.getpixel((997, 2)) == (255, 255, 255))
check("tilt 顶中仍是网格红", t.getpixel((500, 2))[0] > 150)
check("tilt 底角不变", t.getpixel((2, 997)) == (200, 30, 30))
t0 = m.tilt_grid(red, 0)
check("tilt=0 原样", t0.getpixel((2, 2)) == (200, 30, 30))
# 合成链路带 tilt 不炸
out7, _ = m.compose(grid, person_rgba(500, 1000), height_ratio=1.0,
                    center_x=50.0, feet_y=100.0, shadow=True, tilt=10)
check("tilt 合成尺寸不变", out7.size == (1000, 1000))

print()
if FAILED:
    print(f"{FAILED} 项失败")
    sys.exit(1)
print("全部通过")
