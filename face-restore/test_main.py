#!/usr/bin/env python3
"""face-restore 本机契约单测（不依赖 GPU/推理服务/torch/numpy/insightface）。

覆盖服务端算子族（server_op.py = _common/server_ops/face_restore.py 的分发副本）里
纯逻辑部分：F1 版本化注册名、模型权重定位与错误文案、入参校验、裁剪框/蒙版几何。
真机推理路径由 e2e 验收（dev-plan §48.6 A6）。

用法：python3 nodes/face-restore/test_main.py
"""
import importlib.util
import os
import pathlib
import re
import sys
import tempfile

from PIL import Image

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

tmp = tempfile.mkdtemp(prefix="flowx-facerestore-test-")
os.environ["FACERESTORE_MODELS_DIR"] = tmp

spec = importlib.util.spec_from_file_location("sop", HERE / "server_op.py")
sop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sop)

FAILED = 0


def check(label, cond):
    global FAILED
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        FAILED += 1


# ---------- F1 注册契约 ----------
check("OP_VERSION 为 12 位十六进制", re.fullmatch(r"[0-9a-f]{12}", sop.OP_VERSION) is not None)
check("OP_FAMILY == 文件 stem", sop.OP_FAMILY == "face_restore")

captured = {}


class FakeRegistry:
    def register(self, name, inputs, outputs, description=""):
        captured.update(name=name, inputs=inputs, outputs=outputs)
        return lambda fn: captured.setdefault("fn", fn)


sop.register(FakeRegistry())
check("注册名带版本后缀", captured.get("name") == f"face.restore@{sop.OP_VERSION}")
check("输入含 image/model/upscale/padding/feather",
      {"image", "model", "upscale", "padding", "feather"} <= set(captured["inputs"]))
check("输出 = image + faces", captured["outputs"]["image"] == "IMAGE"
      and captured["outputs"]["faces"] == "INT")

# ---------- 模型定位 ----------
try:
    sop._model_path("codeformer", "face.restore")
    check("CodeFormer → ValueError（spandrel 0.4.2 无此架构）", False)
except ValueError as e:
    check("CodeFormer → ValueError（spandrel 0.4.2 无此架构）", "spandrel 0.4.2" in str(e))

try:
    sop._model_path("gfpgan", "face.restore")
    check("权重缺失 → FileNotFoundError 点名目录与文件名", False)
except FileNotFoundError as e:
    msg = str(e)
    check("权重缺失 → FileNotFoundError 点名目录与文件名",
          tmp in msg and "GFPGANv1.3.pth" in msg)

(pathlib.Path(tmp) / "GFPGANv1.3.pth").write_bytes(b"x" * 100)
check("权重齐备 → 返回路径", sop._model_path("gfpgan", "face.restore")
      == os.path.join(tmp, "GFPGANv1.3.pth"))
try:
    sop._model_path("restoreformer", "face.restore")
    check("RestoreFormer 权重缺失 → FileNotFoundError", False)
except FileNotFoundError:
    check("RestoreFormer 权重缺失 → FileNotFoundError", True)

# ---------- 入参校验（在触碰 CUDA 之前）----------
try:
    sop.face_restore("not-an-image")
    check("非 IMAGE 入参 → ValueError", False)
except ValueError as e:
    check("非 IMAGE 入参 → ValueError", "IMAGE 对象" in str(e))

try:
    sop.face_restore(Image.new("RGB", (64, 64)), upscale=9.0)
    check("upscale 越界 → ValueError", False)
except ValueError as e:
    check("upscale 越界 → ValueError", "upscale" in str(e))

try:
    sop.face_restore(Image.new("RGB", (64, 64)), padding=5.0)
    check("padding 越界 → ValueError", False)
except ValueError as e:
    check("padding 越界 → ValueError", "padding" in str(e))

# GFPGAN 512² 限制：模型输入边长必须 512（真机事故：喂 640 报 shapes 不能相乘）
try:
    sop.face_restore(Image.new("RGB", (64, 64)), model="gfpgan", upscale=2.0,
                     det_thresh=0.1)
    check("GFPGAN + upscale=2 → ValueError（512² 固定）", False)
except ValueError as e:
    check("GFPGAN + upscale=2 → ValueError（512² 固定）",
          "upscale" in str(e) or "512" in str(e))
except Exception as e:
    check(f"GFPGAN + upscale=2 → 先报参数错（实际 {type(e).__name__}）", False)

try:
    sop.face_restore(Image.new("RGB", (64, 64)), model="gfpgan")
    check("权重本地存在但 torch/insightface 缺失时不误报参数错", True)
except (FileNotFoundError, ValueError) as e:
    check("权重本地存在但 torch/insightface 缺失时不误报参数错", False)
except Exception:
    check("权重本地存在但 torch/insightface 缺失时不误报参数错", True)

# ---------- 几何：裁剪框 / 蒙版 / 选脸 ----------
check("crop_box 外扩 30% 且被画布裁剪",
      sop._crop_box((100, 100, 200, 200), 0.3, 250, 250) == (70, 70, 230, 230))
check("crop_box 触边钳制到画布",
      sop._crop_box((0, 0, 50, 50), 1.0, 100, 100) == (0, 0, 100, 100))
m = sop._ellipse_mask((64, 64), 0)
check("羽化 0 的椭圆蒙版：中心 255、角点 0（硬边）",
      m.getpixel((32, 32)) == 255 and m.getpixel((0, 0)) == 0)
mf = sop._ellipse_mask((64, 64), 8)
mids = [i for i, c in enumerate(mf.histogram()) if 0 < i < 255]
check("羽化 8 的椭圆蒙版有过渡灰阶", bool(mids))

# 两脸面积相同，位置不对称：一张在左上角、一张靠近中心
faces = [((0, 0, 10, 10), 100.0), ((60, 60, 70, 70), 100.0)]
center = sop._pick(faces, 100, 100, True, 1)
check("only_center 选中离画面中心最近的脸", center[0][0][0] == 60)
biggest = sop._pick(faces, 100, 100, False, 2)
check("非 only_center 时按面积取 max_faces 张", len(biggest) == 2)
check("max_faces 生效（取 1 张）", len(sop._pick(faces, 100, 100, False, 1)) == 1)

print()
print("FAILED =", FAILED)
sys.exit(1 if FAILED else 0)
