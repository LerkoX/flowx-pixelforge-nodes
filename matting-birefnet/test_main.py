#!/usr/bin/env python3
"""matting-birefnet 本机契约单测（不依赖 GPU/推理服务/torch/numpy）。

覆盖服务端算子族（server_op.py = _common/server_ops/matting.py 的分发副本）里
**纯逻辑**部分：F1 版本化注册名、变体目录/文件校验的可操作报错、入参校验。
真正的推理路径由真机 e2e 验收（dev-plan §48.6）。

用法：python3 nodes/matting-birefnet/test_main.py
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

tmp = tempfile.mkdtemp(prefix="flowx-matting-test-")
os.environ["MATTING_MODELS_DIR"] = tmp  # 必须在导入族文件前设好

spec = importlib.util.spec_from_file_location("sop", HERE / "server_op.py")
sop = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sop)

FAILED = 0


def check(label, cond):
    global FAILED
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        FAILED += 1


# ---------- F1：版本戳与注册名 ----------
check("OP_VERSION 为 12 位十六进制（F1 版本戳）",
      re.fullmatch(r"[0-9a-f]{12}", sop.OP_VERSION) is not None)
check("OP_FAMILY == 文件 stem（gate J 合同）", sop.OP_FAMILY == "matting")

captured = {}


class FakeRegistry:
    def register(self, name, inputs, outputs, description=""):
        captured["name"] = name
        captured["inputs"] = inputs
        captured["outputs"] = outputs
        return lambda fn: captured.setdefault("fn", fn)


sop.register(FakeRegistry())
check("注册名带版本后缀（算子身份 = 实现内容）",
      captured.get("name") == f"matting.birefnet@{sop.OP_VERSION}")
check("输入端口含 image/variant/resolution",
      {"image", "variant", "resolution"} <= set(captured.get("inputs", {})))
check("输出端口 = mask + image（灰度 alpha + RGBA）",
      {"mask", "image"} <= set(captured.get("outputs", {})))
check("mask/image 输出类型为 IMAGE", captured["outputs"]["mask"] == "IMAGE"
      and captured["outputs"]["image"] == "IMAGE")

# ---------- 变体目录/文件校验（可操作报错）----------
try:
    sop._variant_dir("nope", "matting.birefnet")
    check("非法 variant → ValueError", False)
except ValueError as e:
    check("非法 variant → ValueError", "variant 非法" in str(e))

try:
    sop._variant_dir("lite", "matting.birefnet")
    check("变体目录缺失 → FileNotFoundError 且点名路径", False)
except FileNotFoundError as e:
    check("变体目录缺失 → FileNotFoundError 且点名路径",
          "lite" in str(e) and tmp in str(e))

os.makedirs(os.path.join(tmp, "lite"))
try:
    sop._variant_dir("lite", "matting.birefnet")
    check("变体目录不全 → FileNotFoundError 列出缺失文件", False)
except FileNotFoundError as e:
    msg = str(e)
    check("变体目录不全 → FileNotFoundError 列出缺失文件",
          "model.safetensors" in msg and "birefnet.py" in msg)

os.makedirs(os.path.join(tmp, "main"))
for f in sop.REQUIRED_FILES:
    (pathlib.Path(tmp) / "main" / f).write_text("x")
check("文件齐备 → 返回目录", sop._variant_dir("main", "matting.birefnet")
      == os.path.join(tmp, "main"))

# ---------- 入参校验（在触碰 torch 之前就报错）----------
try:
    sop.matting_birefnet("not-an-image", variant="lite")
    check("非 IMAGE 入参 → ValueError", False)
except ValueError as e:
    check("非 IMAGE 入参 → ValueError", "IMAGE 对象" in str(e))

try:
    sop.matting_birefnet(Image.new("RGB", (8, 8)), variant="lite", resolution=64)
    check("resolution 越界 → ValueError", False)
except ValueError as e:
    check("resolution 越界 → ValueError", "resolution" in str(e))

# 变体目录不全时，即使入参合法也应在推理前抛 FileNotFoundError（而不是 CUDA 报错）
try:
    sop.matting_birefnet(Image.new("RGB", (8, 8)), variant="portrait")
    check("缺权重时先报 FileNotFoundError（不触碰 CUDA）", False)
except FileNotFoundError:
    check("缺权重时先报 FileNotFoundError（不触碰 CUDA）", True)
except Exception as e:  # torch 未装/无 CUDA 会走到这里 ⇒ 说明校验顺序错了
    check(f"缺权重时先报 FileNotFoundError（实际 {type(e).__name__}）", False)

print()
print("FAILED =", FAILED)
sys.exit(1 if FAILED else 0)
