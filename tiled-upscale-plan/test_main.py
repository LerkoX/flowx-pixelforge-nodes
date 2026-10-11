#!/usr/bin/env python3
"""tiled-upscale-plan 本机契约单测（不依赖 GPU/推理服务/torch/numpy）。

覆盖 server_op.py（= _common/server_ops/tiled.py 的分发副本）里**纯逻辑**部分：
F1 版本化注册名、目录/文件校验的可操作报错、切片网格算法与 plan 生成。
真机推理路径见 dev-plan §49.6 的 e2e 验收。

用法：python3 nodes/tiled-upscale-plan/test_main.py
"""
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
FAILED = 0
PASSED = 0


def check(label, cond, extra=""):
    global FAILED, PASSED
    if cond:
        PASSED += 1
        print(f"PASS {label}" + (f"  {extra}" if extra else ""))
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  {extra}" if extra else ""))


def load_family():
    spec = importlib.util.spec_from_file_location("tiled_family",
                                                  HERE / "server_op.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRegistry:
    def __init__(self):
        self.names = []

    def register(self, name, inputs, outputs, description=""):
        self.names.append(name)
        return lambda fn: fn


def main():
    fam = load_family()
    reg = FakeRegistry()
    fam.register(reg)
    check("F1 注册名带版本后缀",
          sorted(reg.names) == sorted(["tiled.plan@" + fam.OP_VERSION,
                                       "tiled.apply@" + fam.OP_VERSION]),
          str(sorted(reg.names)))
    check("OP_FAMILY == 文件 stem", fam.OP_FAMILY == "tiled", fam.OP_FAMILY)
    check("版本戳非全零（已由 sync-common 打戳）",
          fam.OP_VERSION != "000000000000", fam.OP_VERSION)

    # 一维切分
    check("_split 覆盖到末尾且步长 = tile-overlap",
          fam._split(1000, 384, 64) == [(0, 384), (320, 704), (640, 1000)],
          str(fam._split(1000, 384, 64)))
    check("_split 单块", fam._split(200, 384, 64) == [(0, 200)])

    # 目标尺寸
    tw, th, sc = fam._target_size(512, 512, 2.0, 0, 0, "t")
    check("_target_size 按倍数", (tw, th) == (1024, 1024) and abs(sc - 2.0) < 1e-6,
          f"{tw}x{th} sc={sc}")
    tw, th, sc = fam._target_size(300, 200, 0, 0, 0, "t") if False else (0, 0, 0)
    tw, th, _ = fam._target_size(512, 512, 0, 1000, 700, "t")
    check("_target_size 显式目标尺寸 + 8 对齐", (tw, th) == (1000, 696), f"{tw}x{th}")
    try:
        fam._target_size(512, 512, 0, 0, 0, "t")
        check("_target_size scale<=0 且无目标应报错", False)
    except ValueError:
        check("_target_size scale<=0 报错", True)

    # 网格
    cells = fam._grid(512, 512, 1024, 1024, 768, 96, "t")
    check("_grid 2x2 且序号连续", len(cells) == 4
          and [c["index"] for c in cells] == [0, 1, 2, 3], str(len(cells)))
    check("_grid 坐标 8 对齐", all(c["x"] % 8 == 0 and c["y"] % 8 == 0 for c in cells))
    check("_grid 覆盖整图", all(c["x"] + c["w"] <= 1024 and c["y"] + c["h"] <= 1024
                              for c in cells)
          and any(c["x"] + c["w"] == 1024 for c in cells)
          and any(c["y"] + c["h"] == 1024 for c in cells))
    for bad in ((100, 96), (768, 500), (768, -1)):
        try:
            fam._grid(512, 512, 1024, 1024, bad[0], bad[1], "t")
            check(f"_grid 非法 tile/overlap {bad} 应报错", False)
        except ValueError:
            check(f"_grid 非法 tile/overlap {bad} 报错", True)

    # plan 解析
    import json as _json
    good = _json.dumps({"target": [1024, 1024], "tile": 768, "overlap": 96,
                        "cells": [], "seed": -1, "seam_fix": "mask"})
    check("_parse_plan 正常解析", fam._parse_plan(good, "t")["tile"] == 768)
    for bad, why in (("not-json", "非 JSON"), (_json.dumps({"target": [1, 1]}), "缺字段")):
        try:
            fam._parse_plan(bad, "t")
            check(f"_parse_plan {why} 应报错", False)
        except ValueError as e:
            check(f"_parse_plan {why} 报错", "plan" in str(e) or "字段" in str(e),
                  str(e)[:60])

    # 网格覆盖层（PIL）
    from PIL import Image
    img = Image.new("RGB", (1024, 1024), (30, 30, 30))  # 覆盖层画在目标尺寸上（op 内先缩放）
    ov = fam._overlay_grid(img, fam._grid(512, 512, 1024, 1024, 768, 96, "t"), 768, 96)
    check("_overlay_grid 输出尺寸 = 目标尺寸", ov.size == (1024, 1024), str(ov.size))
    check("_overlay_grid 画上了框线（非纯色）", len(set(ov.getdata())) > 8,
          str(len(set(ov.getdata()))))

    print(f"\nFAILED = {FAILED}   PASSED = {PASSED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
