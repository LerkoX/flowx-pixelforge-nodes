#!/usr/bin/env python3
"""iclight-load 本机契约单测（不依赖 GPU/推理服务/torch/numpy）。

重点覆盖**IC-Light 推理核的完整性**（这是本族最大的维护风险：官方三处关键机制
conv_in 扩通道 / delta 权重相加 / 双条件注入 一旦被误改就会静默出错或精度漂移）
+ 纯逻辑契约（版本化注册名、模型键解析、权重/基座路径校验、尺寸对齐、JSON 信封）。

用法：python3 nodes/iclight-load/test_main.py
"""
import importlib.util
import pathlib
import sys
import tempfile

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
    spec = importlib.util.spec_from_file_location("icl_fam", HERE / "server_op.py")
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
          sorted(reg.names) == sorted([f"iclight.load@{fam.OP_VERSION}",
                                       f"iclight.relight@{fam.OP_VERSION}"]),
          str(sorted(reg.names)))
    check("OP_FAMILY == 文件 stem", fam.OP_FAMILY == "iclight", fam.OP_FAMILY)
    check("版本戳非全零（已由 sync-common 打戳）",
          fam.OP_VERSION != "000000000000", fam.OP_VERSION)
    check("模块导入没 import torch（Termux 本机可跑）", "torch" not in sys.modules)

    # --- 官方三处关键机制（文本级回归：改错会静默出错）---
    check("conv_in 扩通道手术在位（zero → 拷前 4 通道 → 复用 bias）",
          "nc.weight.zero_()" in SRC and "nc.weight[:, :4, :, :].copy_(unet.conv_in.weight)" in SRC
          and "nc.bias = unet.conv_in.bias" in SRC)
    check("IC-Light 权重按 delta 相加（origin + offset）",
          "ori[k] + off[k].to(ori[k].dtype)" in SRC)
    check("权重严格加载（strict=True）", "strict=True" in SRC)
    check("权重键不匹配时大声失败并给人话",
          "权重与基座不匹配" in SRC and "missing=" in SRC and "shape_bad=" in SRC)
    check("双条件注入猴子补丁在位（concat_conds → 通道维拼接）",
          "concat_conds" in SRC and "torch.cat([sample, cc], dim=1)" in SRC
          and "_icl_patched" in SRC)
    check("条件批 1 时按 sample 批扩张（num_images_per_prompt）",
          "sample.shape[0] // cc.shape[0]" in SRC)
    check("官方 run_rmbg(sigma=0) 灰底合成公式在位",
          "127.0 + (src - 127.0) * a[:, :, None]" in SRC)
    check("fc/fbc 梯度区间区分（fc 0~255 / fbc 32~224）",
          '"grad": (0, 255)' in SRC and '"grad": (32, 224)' in SRC)
    check("fbc + none 用官方 64 灰底",
          "np.zeros((h, w, 3), dtype=np.uint8) + 64" in SRC)
    check("fbc 低清恒用 t2i（背景只作条件，官方口径）",
          "fbc：官方**恒用 t2i**" in SRC)
    check("两阶段（低清 + 可选 highres i2i）在位",
          "highres_scale" in SRC and "p[\"denoise_high\"]" in SRC)
    check("同种子显式 generator（逐字节可复现的前提）",
          'manual_seed(int(p["seed"]))' in SRC)
    check("prompt 分块编码（官方 encode_prompt_pair，防 77 token 截断）",
          "def _encode_prompt_pair" in SRC and "model_max_length" in SRC)
    check("进 LRU 记账（复用 model_manager 淘汰/pin/unload 语义）",
          "models._pipes[key] = obj" in SRC and "models._evict_if_needed" in SRC)
    check("精度切换时卸载重建（避免 fp16/fp32 混用）",
          "models.unload(key)" in SRC)
    check("SDPA 注意力处理器（AttnProcessor2_0）", "AttnProcessor2_0" in SRC)
    check("明暗平衡指标（落盘复核判据）在位",
          "light_balance" in SRC and "lum_left" in SRC)

    # --- 变体表自洽 ---
    for v, d in fam.VARIANTS.items():
        check(f"变体 {v} conv_in == 4 + 4*条件数",
              d["conv_in"] == 4 + 4 * d["cond"], f"conv_in={d['conv_in']} cond={d['cond']}")
    check("变体集合 = fc/fbc", sorted(fam.VARIANTS) == ["fbc", "fc"])
    check("光方向集合", fam.LIGHT_DIRS == ("none", "left", "right", "top", "bottom"))

    # --- 纯逻辑 ---
    check("模型键解析正确", fam._key_parts("iclight:fbc", "t") == ("iclight:fbc", "fbc"))
    for bad in ("", "iclight:", "iclight:nope", "ddcolor:modelscope", "fc"):
        try:
            fam._key_parts(bad, "t")
            check(f"非法模型键必须报错：{bad!r}", False, "未报错")
        except ValueError:
            check(f"非法模型键报错：{bad!r}", True)

    check("JSON 信封往返", fam._loads(fam._dumps({"light_balance": 62.0}))
          == {"light_balance": 62.0})
    check("信封前缀是 @", fam._dumps([1, 2]).startswith("@"))
    check("容忍裸 JSON", fam._loads('{"seed": 12345}') == {"seed": 12345})

    for v, ok in ((64, 64), (512, 512), (500, 448), (60, 64), (3000, 2048), (128.9, 128)):
        try:
            got = fam._align64(v, "t", "w")
        except ValueError:
            got = None
        check(f"边长对齐 64：{v} → {got}", got == ok)
    check("_round64 给 0 表示按输入尺寸（返回 0）", fam._round64(0, 0, "t") == (0, 0))
    check("_round64 非 0 时对齐", fam._round64(500, 513, "t") == (448, 512))

    # 权重路径：覆盖路径存在 / 不存在 / 变体非法
    with tempfile.TemporaryDirectory() as td:
        f = pathlib.Path(td) / "w.safetensors"
        f.write_bytes(b"x" * 16)
        check("weights 覆盖路径可用", fam._weights_path("fc", str(f), "t") == str(f))
        try:
            fam._weights_path("fc", str(pathlib.Path(td) / "nope.safetensors"), "t")
            check("weights 覆盖路径不存在必须报错", False, "未报错")
        except FileNotFoundError as e:
            check("weights 覆盖路径不存在报错带路径", "nope.safetensors" in str(e))
    try:
        fam._weights_path("nope", "", "t")
        check("变体非法必须报错", False, "未报错")
    except ValueError:
        check("变体非法报错", True)

    # 基座目录校验（临时目录造 5 个必需项）
    old_base = fam.BASE_DIR
    try:
        with tempfile.TemporaryDirectory() as td:
            for n in ("model_index.json", "unet", "vae", "text_encoder", "tokenizer"):
                d = pathlib.Path(td) / n
                if n.endswith(".json"):
                    d.write_text("{}")
                else:
                    d.mkdir()
            fam.BASE_DIR = td
            check("基座目录完整时通过", fam._base_dir("t") == td)
            (pathlib.Path(td) / "unet").rmdir()
            try:
                fam._base_dir("t")
                check("基座缺件必须报错", False, "未报错")
            except FileNotFoundError as e:
                check("基座缺件报错点名缺失项", "unet" in str(e))
    finally:
        fam.BASE_DIR = old_base

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
