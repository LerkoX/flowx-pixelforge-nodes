#!/usr/bin/env python3
"""gate N：main() 桩冒烟 —— 对每个节点真跑一遍 `main()`。

**为什么需要**（B4/S2 真机事故）：`node mock` / Studio「运行测试」跑的是 `mock.py`，
**跑不到 `main.py`**；而生产执行恰恰只跑 `main.py`。于是下面这类错误只能在真机暴露：
  - B4/S2 `image-watermark`：`NameError: name 'opacity' is not defined`（变量没读进局部变量）
  - B4/S3 `iclight`：可选图像 `mask`/`bg` 漏传/传空导致 `save-image` 参数展开失败
本门禁把 `flowx_client` / `image_io` / `executor_base` 全部换成桩，用 flowx.json 里的
参数默认值喂进去，真跑一遍 `main()`，任何未定义名字、参数名写错、漏传都会在本机暴露。

判定：
  PASS  main() 正常返回
  SKIP  依赖本机没有的库（numpy/torch/cv2…）或已知需要真实服务/文件的节点（见 SKIP_NODES）
  FAIL  其余异常 —— 门禁失败

用法：python3 nodes/_tools/test-main-smoke.py [--verbose] [节点名…]
"""
import importlib.util
import io
import json
import pathlib
import sys
import traceback
import types as _types
from contextlib import redirect_stdout

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 已知需要真实服务/外部文件/GPU、无法桩冒烟的节点（人名 + 原因，不含"我不想修"）
SKIP_NODES = {
    # 直接 urllib/http 访问服务或校验文件系统，不是参数拼装问题
    "inference-ensure": "直接访问推理服务做健康检查（无 run_op 抽象）",
    "inference-ensure-mock": "同上",
}

_MISSING_LIB_HINT = ("numpy", "torch", "cv2", "scipy", "skimage", "onnxruntime",
                     "insightface", "diffusers", "transformers", "timm", "kornia",
                     "einops", "safetensors")


class Recorder:
    def __init__(self):
        self.calls = {}
        self.unknown = set()

    def add(self, key, val=None):
        self.calls.setdefault(key, []).append(val)


# 常见**算子输出键**白名单：桩要"先装满"这些键，节点里
# `out = {k: m.get("id", m.get("value")) for k, m in result["outputs"].items()}`
# 这种字典推导才不会得到空 dict（否则报出假的 KeyError）。
_OP_KEYS = ("model", "clip", "clip2", "vae", "cond", "uncond", "latent", "image",
            "mask", "seed", "info", "width", "height", "count", "ids", "refs",
            "model_ref", "clip_ref", "vae_ref", "t5", "text", "tokens", "score",
            "boxes", "candidates", "sizes", "plan", "meta", "report", "video",
            "frames", "motion", "audio", "list", "detail", "preview",
            "mesh", "num_vertices", "num_faces", "format", "download_url")


# 数值型输出键（其余一律给字符串，模拟 Studio 的 string 端口）
_NUM_KEYS = {"width", "height", "count", "params_mb", "vram_mb", "seed", "steps",
             "current", "total", "size", "batch", "index", "id_num", "sha"}


def _tiny_png():
    """真实可解码的小 PNG（node 的 main 里 Image.open 需要合法字节）。"""
    from PIL import Image
    import io as _io
    buf = _io.BytesIO()
    Image.new("RGB", (64, 48), (10, 20, 30)).save(buf, format="PNG")
    return buf.getvalue()


def _meta_val(key):
    return {"id": f"stub-{key}", "value": f"stub-{key}", "type": "IMAGE"}


def _flat_val(key):
    return 1 if key in _NUM_KEYS else f"stub-{key}"


class StubOut(dict):
    """宽容输出字典：节点只读自己声明的输出键，桩就"任何键都答得到"。

    背景：`run_op`/`call_op`/`wait_job` 的返回值里，键名由节点 flowx.json 的
    outputs 决定；桩若返回空 dict，节点 `out['image']` 就 KeyError，
    会把"桩保真度不足"误报成"节点有 bug"（第一版门禁 41 个假 FAIL 的主因）。
    """

    _self = None

    def __init__(self):
        super().__init__({k: _flat_val(k) for k in _OP_KEYS})
        # ⚠️ 必须**预置** outputs：dict.get("outputs", {}) 不会触发 __missing__，
        # 只靠 __missing__ 时 .get 会返回默认空 dict ⇒ 字典推导得空、报假 KeyError
        self["outputs"] = self
        self["status"] = "success"

    def __missing__(self, key):
        if key == "outputs":
            return self
        if key == "status":
            return "success"
        if key == "job_id":
            return "0" * 16
        return _flat_val(key)


def stub_out(**kw):
    """executor_base.run_op 的返回形状：**已拍平**的 {key: 值}。"""
    d = StubOut()
    d.update(kw)
    d["outputs"] = d
    return d


class StubMeta(dict):
    """服务端**原始**输出形状：每个输出是 {"id"/"value"/"type"} 元数据字典。

    历史节点（checkpoint-loader / flux-* / video-gen / sd3-* 等）直接调
    `flowx_client.submit_job + wait_job`，读的是这种形状（`outs["cond"]["id"]`）；
    新节点走 `executor_base.run_op` 读拍平后的值。桩必须两种都给对。
    """

    def __init__(self):
        super().__init__({k: _meta_val(k) for k in _OP_KEYS})

    def __missing__(self, key):
        return _meta_val(key)


class StubRaw(dict):
    """原始 API 顶层返回：{"outputs": {k: 元数据}, "status": "success", ...}。"""

    _LISTS = {"resident_models", "ops", "models", "items", "list", "details"}

    def __init__(self):
        super().__init__()
        # 同上：预置 outputs/status，保证 .get("outputs", {}) 拿到真对象
        self["outputs"] = StubMeta()
        self["status"] = "success"

    def __missing__(self, key):
        if key == "outputs":
            return StubMeta()
        if key == "status":
            return "success"
        if key in self._LISTS:
            return []
        return StubMeta()


def stub_raw(**kw):
    d = StubRaw()
    d.update(kw)
    return d


def _default_value(p, tmpdir=None):
    """按 flowx.json 声明与默认值给出注入串（Studio 的 env 注入一律是字符串）。

    空串/缺省值给非空桩：很多节点有"至少要给一个"的自校验（如 image-watermark 的
    text/logo），给空串会走校验分支而不是主流程，那样冒烟就白跑了。
    `*_path` / `file*` 参数给一个**真实存在**的临时 PNG（load-image 等要读文件）。
    """
    name = str(p.get("name") or "")
    d = p.get("default")
    if d is not None and d != "":
        if isinstance(d, bool):
            return "true" if d else "false"
        return str(d)
    t = str(p.get("type", "string")).lower()
    if t in ("integer", "number", "float"):
        return "1"
    if t == "boolean":
        return "false"
    low = name.lower()
    if tmpdir is not None and (low.endswith("_path") or low in ("path", "file")):
        from PIL import Image
        f = pathlib.Path(tmpdir) / f"{low or 'file'}.png"
        if not f.exists():
            Image.new("RGB", (64, 48), (30, 40, 50)).save(f)
        return str(f)
    if low in ("text", "prompt", "label", "title", "watermark"):
        return "stub-text"
    return "stub-obj"


def make_stubs(node_dir, params, rec, verbose):
    def _param(name, default=None, cast=None):
        v = params.get(name, default)
        if v is None:
            v = default
        if cast is not None:
            try:
                return cast(v)
            except Exception:
                return cast(0) if cast in (int, float) else v
        return v

    fc = _types.ModuleType("flowx_client")

    def _fallback(name):
        def _fn(*a, **kw):
            rec.add("flowx_client." + name, (a, kw))
            if verbose:
                print(f"    [stub] flowx_client.{name}({a}, {kw})")
            if "bytes" in name:
                return _tiny_png()
            if name.startswith("submit") or name.endswith("_job"):
                return "0" * 16          # 节点会对 jid 做 jid[:8]
            if name.startswith(("get_", "post_", "put_", "wait_", "call_")):
                return stub_raw()
            return None
        return _fn

    def _fc_getattr(name):
        rec.unknown.add("flowx_client." + name)
        return _fallback(name)

    fc.__getattr__ = _fc_getattr
    fc.call_op = lambda *a, **kw: stub_raw()
    fc.submit_job = lambda *a, **kw: "0" * 16
    fc.wait_job = lambda *a, **kw: stub_raw()
    fc.get_json = lambda *a, **kw: stub_raw()
    fc.post_json = lambda *a, **kw: stub_raw()
    fc.get_bytes = lambda *a, **kw: _tiny_png()
    # 上传应答：/images 回 {"id"}，/videos 回 {"path"}（POST /videos 只做字节入站，
    # 再把服务端路径交给 video.load 算子）——两种键都给，桩才对得上真实服务
    fc.post_bytes = lambda *a, **kw: {"id": "stub-obj",
                                      "path": "/tmp/stub-upload.mp4"}
    fc.put_bytes = lambda *a, **kw: {"id": "stub-obj"}
    fc.param = _param
    fc.token = lambda: "stub-token"
    fc.ref = lambda oid: {"$id": oid}
    fc.emit = lambda **kw: rec.add("emit", kw)
    fc.emit_preview = lambda *a, **kw: rec.add("emit_preview", (a, kw))
    fc.emit_result_preview = lambda *a, **kw: rec.add("emit_result_preview", (a, kw))
    fc.yaml_scalar = json.dumps
    fc.host_base = lambda u: u
    sys.modules["flowx_client"] = fc

    io_mod = _types.ModuleType("image_io")

    def _pimg(base, oid, tok=None, mode=None):
        from PIL import Image
        rec.add("get_pimg", (base, oid, mode))
        if str(mode).upper() == "L":
            return Image.new("L", (64, 48), 128)
        return Image.new("RGBA" if str(mode).upper() == "RGBA" else "RGB",
                         (64, 48), (10, 20, 30, 255))

    io_mod.get_pimg = _pimg
    io_mod.post_pimg = lambda base, pil, tok=None: (rec.add("post_pimg", pil.size),
                                                   "obj-out")[1]
    io_mod.pbool = lambda raw: str(raw).strip().lower() in ("1", "true", "yes", "on")
    io_mod.RESAMPLE = {"nearest": 0, "bilinear": 2, "bicubic": 3, "lanczos": 1}
    io_mod.save_pimg = lambda *a, **kw: rec.add("save_pimg", a)

    def _io_getattr(name):
        rec.unknown.add("image_io." + name)
        return _fallback("image_io." + name)

    io_mod.__getattr__ = _io_getattr
    sys.modules["image_io"] = io_mod

    eb = _types.ModuleType("executor_base")
    eb.run_op = lambda op, inputs, emit_keys=(), **kw: (rec.add("run_op", {
        "op": op, "inputs": inputs, "emit_keys": list(emit_keys)}),
        stub_out(**{k: f"stub-{k}" for k in emit_keys}))[1]
    eb.ensure_op = lambda url, tok, op: op
    eb.host_base = lambda u: u

    def _eb_getattr(name):
        rec.unknown.add("executor_base." + name)
        return _fallback("executor_base." + name)

    eb.__getattr__ = _eb_getattr
    sys.modules["executor_base"] = eb


# emit 输出里禁止出现的字符（Studio 侧 dag/extractor.go 的 extractor 陷阱）：
#   removeComments 用 `\s*#.*$` 无差别删行尾注释 ⇒ 引号里的 #RRGGBB 也被截断
#   ⇒ 整块 YAML 解析失败 ⇒ 该节点**所有输出丢失**（image-shadow exec 714 实测踩到）；
#   闭合围栏 ``` 出现在值里会提前闭合代码块（exec 158/159 事故）。
_FORBIDDEN_IN_EMIT = ("#", "```")


def _forbidden_emits(rec):
    bad = []
    for kw in rec.calls.get("emit", []):
        if not isinstance(kw, dict):
            continue
        for k, v in kw.items():
            if isinstance(v, str):
                for needle in _FORBIDDEN_IN_EMIT:
                    if needle in v:
                        bad.append((k, needle, v[:60]))
    return bad


def smoke(node_dir, verbose=False):
    """返回 (status, detail)：PASS / SKIP / FAIL。"""
    fj = json.loads((node_dir / "flowx.json").read_text())
    name = fj.get("name") or node_dir.name
    if name in SKIP_NODES:
        return "SKIP", SKIP_NODES[name]
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        params = {p["name"]: _default_value(p, td)
                  for p in fj.get("parameters") or []}
        return _run_node(node_dir, name, params, verbose)


def _run_node(node_dir, name, params, verbose):
    rec = Recorder()
    make_stubs(node_dir, params, rec, verbose)
    # 真机执行时节点目录就是工作目录 ⇒ 必须进 sys.path（image_filters 等靠它）
    sys.path.insert(0, str(node_dir))
    spec = importlib.util.spec_from_file_location(
        f"smoke_{node_dir.name.replace('-', '_')}", node_dir / "main.py")
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            spec.loader.exec_module(mod)
            fn = getattr(mod, "main", None)
            if fn is None:
                return "FAIL", "main.py 里没有 main()"
            fn()
    except (ModuleNotFoundError, ImportError) as e:
        miss = str(e).split("'")[1] if "'" in str(e) else str(e)
        if any(h in miss for h in _MISSING_LIB_HINT):
            return "SKIP", f"本机缺库 {miss}（推理镜像里有）"
        return "FAIL", f"{type(e).__name__}: {e}"
    except (ValueError, RuntimeError, AssertionError) as e:
        # 节点自身对桩输入的显式校验（"text 与 logo 至少要给一个"之类）
        # —— 冒烟的意义是抓未定义名字/参数拼错，这类归 SKIP 并打印原因
        return "SKIP", f"节点自校验拦下桩输入：{type(e).__name__}: {str(e)[:120]}"
    except Exception as e:
        tb = traceback.format_exc().strip().splitlines()
        tail = " | ".join(x.strip() for x in tb[-4:-1])[:300]
        return "FAIL", f"{type(e).__name__}: {e}  [{tail}]"
    else:
        bad = _forbidden_emits(rec)
        if bad:
            k, needle, val = bad[0]
            return "FAIL", (f"emit 值 '{k}' 含 {needle!r} ⇒ Studio codec-block 提取器"
                            f"会截断/提前闭合 ⇒ 该节点输出全部丢失（值前 60 字：{val}）")
    finally:
        for m in ("flowx_client", "image_io", "executor_base"):
            sys.modules.pop(m, None)
        if str(node_dir) in sys.path:
            sys.path.remove(str(node_dir))
    return "PASS", ""


NEG_CASES = {
    # 负例名字: (main.py 源码, 期望判定)
    "nameerror": ("from flowx_client import param\n\n\n"
                  "def main():\n"
                  "    print(opacity)   # 未定义名字（B4/S2 真机事故原型）\n", "FAIL"),
    "typeerror": ("from flowx_client import param\n\n\n"
                  "def main():\n"
                  "    param('p').split(1, 2)  # 参数类型/个数写错（真实 TypeError）\n",
                  "FAIL"),
    "attributeerror": ("from flowx_client import param\n\n\n"
                       "def main():\n"
                       "    param('v').no_such_method()  # 类型/属性写错\n",
                       "FAIL"),
    "okcase": ("from flowx_client import emit, param\n\n\n"
               "def main():\n"
               "    emit(image='x', width=1, height=1)\n", "PASS"),
    "guardcase": ("from flowx_client import param\n\n\n"
                  "def main():\n"
                  "    raise ValueError('参数校验：至少要给一个')\n", "SKIP"),
    # 平台陷阱负例：emit 值带 # ⇒ 提取器删注释后 YAML 块解析失败、输出全丢
    "hashvalue": ("from flowx_client import emit\n\n\n"
                  "def main():\n"
                  "    emit(image='x', info='{\"color\": \"#101014\"}')\n",
                  "FAIL"),
    "fencevalue": ("from flowx_client import emit\n\n\n"
                   "def main():\n"
                   "    emit(image='x', summary='a ``` b')\n", "FAIL"),
}


def selftest():
    """负例自检：门禁必须真的抓得住 main.py 里的错误，否则形同虚设。"""
    import tempfile
    ok = 0
    total = 0
    with tempfile.TemporaryDirectory() as td:
        for name, (src, want) in NEG_CASES.items():
            d = pathlib.Path(td) / ("selftest-" + name)
            d.mkdir()
            (d / "main.py").write_text(src)
            (d / "flowx.json").write_text(json.dumps(
                {"name": "selftest-" + name, "entry": "main.py", "files": [],
                 "parameters": [{"name": "p", "type": "string", "default": "v"}],
                 "outputs": [{"name": "image", "type": "string"}]}))
            got, detail = _run_node(d, d.name, {"p": "v"}, False)
            total += 1
            hit = (got == want)
            ok += 1 if hit else 0
            print(f"  {'PASS' if hit else 'FAIL'} 负例 {name}: 期望 {want} 实得 {got}"
                  + (f" ({detail[:60]})" if detail else ""))
    print(f"\n门禁自检：{ok}/{total} 符合预期")
    return 0 if ok == total else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    argv = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    dirs = sorted([d for d in ROOT.iterdir()
                   if (d / "flowx.json").is_file() and (d / "main.py").is_file()
                   and (not argv or d.name in argv)], key=lambda d: d.name)
    passed, skipped, failed = [], [], []
    for d in dirs:
        st, detail = smoke(d, verbose=verbose)
        if st == "PASS":
            passed.append(d.name)
        elif st == "SKIP":
            skipped.append((d.name, detail))
        else:
            failed.append((d.name, detail))
            print(f"FAIL {d.name}: {detail}")
    print(f"\nmain() 桩冒烟：PASS {len(passed)} / SKIP {len(skipped)} / FAIL {len(failed)}"
          f"（共 {len(dirs)} 个节点）")
    if skipped:
        print("  跳过：")
        for n, why in skipped:
            print(f"    - {n}: {why}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
