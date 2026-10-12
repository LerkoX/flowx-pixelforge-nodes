#!/usr/bin/env python3
"""colorize 本机契约单测（不依赖 GPU/推理服务/torch/numpy）。

重点覆盖**vendored DDColor 实现的完整性**（这是本族最大的维护风险：上游 5 个模块被
内联进惰性工厂，一旦有人误删/漏搬迁就会静默出错）+ 纯逻辑契约（版本化注册名、
模型键解析、JSON 信封、内部边长校验）。

用法：python3 nodes/colorize/test_main.py
"""
import importlib.util
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
FAILED = 0
PASSED = 0
SRC = (HERE / "server_op.py").read_text()


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
    check("惰性工厂存在且未展开", hasattr(fam, "_ddclasses") and fam._CLASSES is None)
    check("模块导入没 import torch", "torch" not in sys.modules)

    # --- vendored 完整性（文本级回归：漏搬迁会静默出错）---
    for banner in ("unet.py", "convnext.py", "transformer_utils.py",
                   "position_encoding.py", "ddcolor/model.py"):
        check(f"vendored 段落标注在位：{banner}", f"===== {banner}" in SRC
              or banner in SRC)
    for cls in ("class Hook", "class CustomPixelShuffle_ICNR",
                "class UnetBlockWide", "class ConvNeXt", "class LayerNorm",
                "class SelfAttentionLayer", "class CrossAttentionLayer",
                "class MLP", "class PositionEmbeddingSine", "class DDColor",
                "class ImageEncoder", "class DuelDecoder",
                "class MultiScaleColorDecoder"):
        check(f"vendored 类在位：{cls}", cls in SRC)
    real_import = [ln for ln in SRC.splitlines()
                   if ln.lstrip().startswith(("from basicsr", "import basicsr"))]
    check("无残留 basicsr 导入语句（必须已内联；散文引用不算）",
          not real_import, str(real_import[:2]))
    check("timm 回退保留（convnext 纯 torch fallback）",
          "timm.layers" in SRC and "def trunc_normal_" in SRC)
    check("convnext-t / convnext-l 两档编码器都在",
          "depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]" in SRC
          and "depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536]" in SRC)
    check("保留原图 L 通道（只预测 ab ⇒ 结构不变）",
          "orig_l = orig_lab[:, :, :1]" in SRC and "COLOR_BGR2Lab" in SRC
          and "COLOR_LAB2BGR" in SRC)
    check("越界 clip 修复在位（上游 astype(uint8) 会回绕）",
          "np.clip(out_bgr" in SRC)
    check("Lab2RGB 灰度复制 L 的输入构造在位（上游口径）",
          "gray_lab = np.concatenate((small_l" in SRC
          and "COLOR_LAB2RGB" in SRC)
    check("权重键不匹配时大声失败（dev-plan §50.7 风险项）",
          "权重与 vendored DDColor 架构不匹配" in SRC)
    check("进 LRU 记账（复用 model_manager 淘汰/pin/unload 语义）",
          "models._pipes[key] = obj" in SRC and "models._evict_if_needed" in SRC)

    check("模型键解析正确", fam._key_parts("ddcolor:artistic", "t")
          == ("ddcolor:artistic", "artistic"))
    for bad in ("", "ddcolor:", "ddcolor:nope", "colorize:modelscope"):
        try:
            fam._key_parts(bad, "t")
            check(f"非法模型键必须报错：{bad!r}", False, "未报错")
        except ValueError:
            check(f"非法模型键报错：{bad!r}", True)

    check("JSON 信封往返", fam._loads(fam._dumps({"chroma": 21.4}))
          == {"chroma": 21.4})
    check("信封前缀是 @", fam._dumps([1, 2]).startswith("@"))
    check("容忍裸 JSON", fam._loads('{"size": 512}') == {"size": 512})

    for s, ok in ((512, True), (256, True), (128, True), (1536, True),
                  (129, False), (512.5, True), (2000, False)):
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
