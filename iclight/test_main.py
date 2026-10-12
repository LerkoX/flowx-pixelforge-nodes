#!/usr/bin/env python3
"""iclight 本机契约单测（不依赖 GPU/推理服务/torch/numpy）。

覆盖：①族契约（版本化注册名 / 惰性 import）②推理参数契约（校验与默认值）
③**main() 桩冒烟**（桩掉 flowx_client/executor_base，真跑一遍 main()，把
"参数名写错 / 漏传可选图像 / run_op 参数拼错"这类只能在真机暴露的错误拦在本机）。

用法：python3 nodes/iclight/test_main.py
"""
import importlib.util
import pathlib
import sys
import types as _types

HERE = pathlib.Path(__file__).parent
FAILED = 0
PASSED = 0
SRC = (HERE / "server_op.py").read_text()
MAIN_SRC = (HERE / "main.py").read_text()


def check(label, cond, extra=""):
    global FAILED, PASSED
    if cond:
        PASSED += 1
        print(f"PASS {label}" + (f"  {extra}" if extra else ""))
    else:
        FAILED += 1
        print(f"FAIL {label}" + (f"  {extra}" if extra else ""))


def load_family():
    spec = importlib.util.spec_from_file_location("icl_fam2", HERE / "server_op.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeRegistry:
    def __init__(self):
        self.names = []

    def register(self, name, inputs, outputs, description=""):
        self.names.append(name)
        return lambda fn: fn


def check_main_smoke():
    """main() 桩冒烟：桩掉 flowx_client / executor_base，真跑一遍 main()。

    为什么必须有：`node mock` 跑的是 mock.py，**跑不到 main.py**（B4/S2 就因此把一个
    `NameError` 带到真机）。这里把 main() 完整跑一次，未定义名字 / 参数名写错 /
    可选图像漏传都会在本机暴露。
    """
    calls = {}

    def _param(name, default=None, cast=None):
        vals = {"service_url": "http://stub", "image": "obj-fg", "model": "iclight:fc",
                "mask": "", "bg": "", "light_dir": "left",
                "prompt": "sunshine from window",
                "negative_prompt": "lowres", "a_prompt": "best quality",
                "width": "768", "height": "0", "steps": "20", "cfg": "2.0",
                "seed": "999", "denoise_low": "0.9", "denoise_high": "0.5",
                "highres_scale": "1.5", "highres": "true", "scheduler": "ddim"}
        v = vals.get(name, default)
        return cast(v) if cast else v

    def _ref(oid):
        return {"$id": oid}

    def _run_op(op, inputs, emit_keys=(), check_exists=False, timeout=3600):
        calls["op"] = op
        calls["inputs"] = inputs
        calls["emit_keys"] = list(emit_keys)
        return {}

    fc = _types.ModuleType("flowx_client")
    fc.param = _param
    fc.ref = _ref
    fc.token = lambda: "tk"
    sys.modules["flowx_client"] = fc
    eb = _types.ModuleType("executor_base")
    eb.run_op = _run_op
    sys.modules["executor_base"] = eb

    spec = importlib.util.spec_from_file_location("icl_main", HERE / "main.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.main()
        check("main() 桩冒烟无异常（名字未定义/参数写错都会在此暴露）", True)
    except Exception as e:
        check("main() 桩冒烟无异常", False, f"{type(e).__name__}: {e}")
        return
    check("main() 调用的算子名正确（基名，ensure_op 会补版本）",
          calls.get("op") == "iclight.relight", str(calls.get("op")))
    ins = calls.get("inputs", {})
    check("image 按对象引用包装", ins.get("image") == {"$id": "obj-fg"}, str(ins.get("image")))
    check("可选的 mask/bg 留空时**不传**（服务端取 None）",
          "mask" not in ins and "bg" not in ins)
    check("数值参数已按类型转换（width=768 int / cfg=2.0 float / seed=999 int）",
          ins.get("width") == 768 and ins.get("cfg") == 2.0 and ins.get("seed") == 999,
          str({k: ins.get(k) for k in ("width", "cfg", "seed")}))
    check("highres/scheduler 透传", ins.get("highres") == "true"
          and ins.get("scheduler") == "ddim")
    check("emit_keys 覆盖落盘判据字段（light_balance/lum_*）",
          {"image", "width", "height", "light_balance", "lum_left", "lum_right",
           "info"} <= set(calls.get("emit_keys", [])))

    # 再跑一次：mask/bg 非空时必须传引用（可选图像的正向路径）
    calls.clear()

    def _param2(name, default=None, cast=None):
        if name == "mask":
            return "obj-mask"
        if name == "bg":
            return "obj-bg"
        return _param(name, default, cast)

    fc.param = _param2
    spec2 = importlib.util.spec_from_file_location("icl_main2", HERE / "main.py")
    mod2 = importlib.util.module_from_spec(spec2)
    try:
        spec2.loader.exec_module(mod2)
        mod2.main()
        ins2 = calls.get("inputs", {})
        check("mask/bg 非空时按引用传入",
              ins2.get("mask") == {"$id": "obj-mask"}
              and ins2.get("bg") == {"$id": "obj-bg"}, str(ins2.get("mask")))
    except Exception as e:
        check("mask/bg 非空路径可运行", False, f"{type(e).__name__}: {e}")


def main():
    fam = load_family()
    reg = FakeRegistry()
    fam.register(reg)
    check("F1 注册名带版本后缀且基名齐全",
          sorted(reg.names) == sorted([f"iclight.load@{fam.OP_VERSION}",
                                       f"iclight.relight@{fam.OP_VERSION}"]),
          str(sorted(reg.names)))
    check("模块导入没 import torch（Termux 本机可跑）", "torch" not in sys.modules)

    # --- 推理参数契约（文本级 + 常量级）---
    check("light_dir 白名单校验在位", "if light not in LIGHT_DIRS" in SRC)
    check("scheduler 只允许 dpmpp/ddim", '("dpmpp", "ddim")' in SRC)
    check("steps/cfg/denoise/highres_scale 都有范围校验",
          "steps 非法" in SRC and "cfg 非法" in SRC and "denoise 非法" in SRC
          and "highres_scale 非法" in SRC)
    check("fc 收到 bg 时给提示（不静默）", "bg 已忽略" in SRC)
    check("mask 可选：没给就不做灰底合成",
          "_gray_composite(np, fg_pil, mask) if mask is not None" in SRC
          and "else np.asarray(fg_pil)" in SRC)
    check("mask 过小时退回全图统计（避免除零/空集）",
          "int(mm.sum()) < 100" in SRC)
    check("不在模块级强引用 Bundle（否则 LRU 淘汰释放不了显存）",
          "_BUNDLES" not in SRC and "唯一真相源是 `model_manager._pipes`" in SRC)
    check("输出含 light_balance/lum_left/lum_right（落盘判据）",
          '"light_balance"' in SRC and '"lum_left"' in SRC and '"lum_right"' in SRC)
    check("main.py 里可选的 mask/bg 走条件传参",
          MAIN_SRC.count("if mask_id:") == 1 and MAIN_SRC.count("if bg_id:") == 1)
    check("main.py 传的算子名与族内注册名基名一致",
          'run_op("iclight.relight"' in MAIN_SRC)

    check_main_smoke()

    print(f"\nPASSED = {PASSED} / FAILED = {FAILED}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
