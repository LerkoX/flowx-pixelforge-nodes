#!/usr/bin/env python3
"""grounding-dino-load 本机契约单测（不依赖 GPU/推理服务/torch/numpy）。

覆盖 server_op.py（= _common/server_ops/seg.py 的分发副本）里**纯逻辑**部分：
F1 版本化注册名、目录/文件校验的可操作报错、装载参数/变体目录校验。
真机推理路径见 dev-plan §49.6 的 e2e 验收。

用法：python3 nodes/grounding-dino-load/test_main.py
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
    spec = importlib.util.spec_from_file_location("seg_family",
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
    check("F1 注册名带版本后缀且基名齐全",
          sorted(reg.names) == sorted([
              "seg.sam_load@" + fam.OP_VERSION, "seg.sam@" + fam.OP_VERSION,
              "seg.sam_auto@" + fam.OP_VERSION,
              "seg.grounding_load@" + fam.OP_VERSION,
              "seg.grounding_detect@" + fam.OP_VERSION]),
          str(sorted(reg.names)))
    check("OP_FAMILY == 文件 stem", fam.OP_FAMILY == "seg", fam.OP_FAMILY)
    check("版本戳非全零（已由 sync-common 打戳）",
          fam.OP_VERSION != "000000000000", fam.OP_VERSION)

    # 提示解析
    pts, labels = fam._parse_points("10,20;30,40,0; 50 , 60 , 1 ", "t")
    check("_parse_points 基本解析", pts == [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]]
          and labels == [1, 0, 1], f"{pts} {labels}")
    check("_parse_points 空串 → 空", fam._parse_points("", "t") == ([], []))
    for bad in ("1", "a,b", "1,2,7"):
        try:
            fam._parse_points(bad, "t")
            check(f"_parse_points 非法 {bad!r} 应报错", False)
        except ValueError:
            check(f"_parse_points 非法 {bad!r} 报错", True)
    check("_parse_box 基本解析", fam._parse_box("1,2,30,40", "t") == [1, 2, 30, 40])
    check("_parse_box 空 → None", fam._parse_box("", "t") is None)
    for bad in ("1,2,3", "10,2,1,40"):
        try:
            fam._parse_box(bad, "t")
            check(f"_parse_box 非法 {bad!r} 应报错", False)
        except ValueError:
            check(f"_parse_box 非法 {bad!r} 报错", True)

    # 模型键解析
    check("_parse_key 接受 'sam:vit_l' 与裸 'vit_l'",
          fam._parse_key("sam:vit_l", "sam", fam.SAM_VARIANTS, "t") == "vit_l"
          and fam._parse_key("vit_l", "sam", fam.SAM_VARIANTS, "t") == "vit_l")
    try:
        fam._parse_key("", "sam", fam.SAM_VARIANTS, "t")
        check("_parse_key 空应报错", False)
    except ValueError:
        check("_parse_key 空报错", True)

    # keep 解析
    check("_parse_keep 空 = 全选", fam._parse_keep("", 3, "t") == [0, 1, 2])
    check("_parse_keep 逗号/分号/空格混用", fam._parse_keep("0, 2;1", 3, "t") == [0, 2, 1])
    try:
        fam._parse_keep("5", 3, "t")
        check("_parse_keep 越界应报错", False)
    except ValueError:
        check("_parse_keep 越界报错", True)

    # 变体目录校验（可操作报错）
    try:
        fam._variant_dir("/nonexistent-dir-b3", "vit_l", fam.SAM_VARIANTS, "t")
        check("变体目录缺失应报错", False)
    except FileNotFoundError as e:
        check("变体目录缺失报错且含下载指引",
              "facebook/sam-vit-large" in str(e), str(e)[:80])
    try:
        fam._variant_dir("/nonexistent-dir-b3", "vit_x", fam.SAM_VARIANTS, "t")
        check("非法变体应报错", False)
    except ValueError as e:
        check("非法变体报错可选值", "vit_h" in str(e), str(e)[:70])

    print(f"\nFAILED = {FAILED}   PASSED = {PASSED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
