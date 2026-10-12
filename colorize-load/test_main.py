#!/usr/bin/env python3
"""colorize-load 本机契约单测（不依赖 GPU/推理服务/torch/numpy）。

覆盖 server_op.py（= `_common/server_ops/colorize.py` 的分发副本）里**纯逻辑**部分：
F1 版本化注册名、惰性工厂（模块导入不得 import torch）、变体目录/权重校验的可操作报错、
模型键解析、JSON 信封、内部推理边长校验。真机推理路径见 dev-plan §50.5 的 e2e 验收。

用法：python3 nodes/colorize-load/test_main.py
"""
import importlib.util
import pathlib
import sys
import tempfile

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
    spec = importlib.util.spec_from_file_location("colorize_family",
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
          sorted(reg.names) == sorted([f"colorize.load@{fam.OP_VERSION}",
                                       f"colorize.apply@{fam.OP_VERSION}"]),
          str(sorted(reg.names)))
    check("OP_FAMILY == 文件 stem", fam.OP_FAMILY == "colorize", fam.OP_FAMILY)
    check("版本戳非全零（已由 sync-common 打戳）",
          fam.OP_VERSION != "000000000000", fam.OP_VERSION)
    check("惰性工厂：模块导入没 import torch（Termux 可测的前提）",
          "torch" not in sys.modules and fam._CLASSES is None)
    check("变体表齐全（4 个）",
          sorted(fam.VARIANTS) == ["artistic", "modelscope", "paper", "paper_tiny"],
          str(sorted(fam.VARIANTS)))
    check("每个变体都带上游仓库名（报错给下载指引）",
          all(v.get("repo", "").startswith("piddnad/") for v in fam.VARIANTS.values()))
    check("必备文件 = config.json + pytorch_model.bin",
          tuple(fam.REQUIRED_FILES) == ("config.json", "pytorch_model.bin"),
          str(fam.REQUIRED_FILES))

    with tempfile.TemporaryDirectory() as td:
        base = pathlib.Path(td)
        (base / "modelscope").mkdir()
        (base / "modelscope" / "config.json").write_text("{}")
        try:
            fam._variant_dir(str(base), "modelscope", "t")
            check("缺 pytorch_model.bin 必须报错", False, "未报错")
        except FileNotFoundError as e:
            check("缺权重报错点名文件与上游仓库",
                  "pytorch_model.bin" in str(e) and "ddcolor_modelscope" in str(e))
        (base / "modelscope" / "pytorch_model.bin").write_bytes(b"x")
        check("文件齐全后返回变体目录",
              fam._variant_dir(str(base), "modelscope", "t")
              == str(base / "modelscope"))
        try:
            fam._variant_dir(str(base), "nope", "t")
            check("非法变体必须报错", False, "未报错")
        except ValueError as e:
            check("非法变体报错列出可选值", "modelscope" in str(e), str(e)[:60])
        try:
            fam._variant_dir(str(base / "missing"), "modelscope", "t")
            check("缺目录必须报错", False, "未报错")
        except FileNotFoundError as e:
            check("缺目录报错含期望路径",
                  "missing/modelscope" in str(e).replace("\\", "/"))

    check("模型键解析正确", fam._key_parts("ddcolor:paper_tiny", "t")
          == ("ddcolor:paper_tiny", "paper_tiny"))
    for bad in ("", "sam:vit_l", "ddcolor:nope"):
        try:
            fam._key_parts(bad, "t")
            check(f"非法模型键必须报错：{bad!r}", False, "未报错")
        except ValueError:
            check(f"非法模型键报错：{bad!r}", True)

    check("JSON 信封往返", fam._loads(fam._dumps({"a": 1, "b": "中文"}))
          == {"a": 1, "b": "中文"})
    check("信封前缀是 @（防 Studio 自动反序列化成 map）",
          fam._dumps({"a": 1}).startswith("@"))
    check("容忍裸 JSON（手工在 UI 填值）", fam._loads('{"a": 1}') == {"a": 1})
    check("空串返回默认值", fam._loads("", "d") == "d")

    for s, ok in ((512, True), (256, True), (1024, True), (128, True),
                  (1536, True), (127, False), (1000, False), (1537, False)):
        try:
            fam._check_size(s, "t")
            got = True
        except ValueError:
            got = False
        check(f"内部边长校验 {s} → {'接受' if ok else '拒绝'}", got == ok)

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
