#!/usr/bin/env python3
"""gate O：服务端算子「声明 ↔ 函数签名 ↔ 节点实参」一致性（纯 AST，零 import、零依赖）。

**为什么需要**（dev-plan P1-7）：节点侧的参数漂移已有 gate M（flowx.json ↔ main.py）与
gate N（`main()` 桩冒烟）挡住；**服务端算子**既没有冒烟、也没有签名校验，下面五类漂移
只能到真机才暴露：

| # | 漂移 | 后果 |
| --- | --- | --- |
| E1 | 声明的 input 端口不在算子函数签名上 | 引擎按 `op.inputs` 的名字传 kwarg ⇒ 真机 `TypeError: unexpected keyword argument` |
| E2 | 算子函数的**必填参数**（无默认值）没进声明 | 引擎只按 `op.inputs` 组装 kwarg ⇒ 真机 `TypeError: missing required argument` |
| E3 | 节点传了**声明里没有**的键 | `resolve_from_ids/resolve_in_process` 只遍历 `op.inputs` ⇒ **参数被静默丢弃**：界面能填、流程 success、值没生效（最阴的一类） |
| E4 | 节点漏传函数的必填参数 | 真机 `TypeError: missing argument`（E2 的另一种触法） |
| E5 | 声明的 output 端口不在函数的 `return` 字典里 | 运行时 `did not return output '<port>'` |
| E6 | 同一算子名出现在两个服务端插件文件里 | 上传必 409（跨文件重名，历史事故） |

**覆盖面**（两类服务端算子，都查）：
1. 族文件 `_common/server_ops/*.py`（sync-common 分发，15 族 / 51 算子 / 51 成员节点）；
2. **遗留自带** `<节点>/server_op.py`（未纳入 sync-common 的 24 个，历史遗留但仍在跑）。

**为什么是"静态比对"而不是"import 后桩冒烟"**：族文件里 5 个（`cond_latent`/`flux`/
`instantid`/`ipadapter`/`vton`）**顶层 `import torch`**，`video_sample`/`ksampler` 等顶层
`from app import ...` —— 本机（手机 Termux，无 GPU/无 torch）import 不了，桩冒烟只能覆盖
少数文件。声明与调用契约的对齐**不需要执行**，AST 足够，且 100% 可复现、零依赖。

判定：E1~E6 任一命中即 FAIL；解析不出来（`inputs=` 动态构造、`args[k]` 动态键、算子名来自
条件分支……）打 WARN 不 FAIL —— 不假装查过，也不误杀。

用法：python3 nodes/_tools/check-op-signatures.py [--selftest] [--verbose]
退出码：0 一致；1 有漂移（逐条打印人话）
"""
from __future__ import annotations

import ast
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 节点侧调用算子的三种形态（老节点用 submit_job，新节点用 run_op / call_op）
CALL_FORMS = ("run_op", "call_op", "submit_job")


# ---------- 单一事实源：族 ↔ 成员节点（复用 sync-common.py，避免两处维护） ----------

def load_families() -> dict[str, list[str]]:
    """从 sync-common.py 读 OP_FAMILIES（唯一事实源）。键是文件名，stem 才是族名。"""
    src = (ROOT / "_tools" / "sync-common.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, "id", "") == "OP_FAMILIES" for t in node.targets):
            return {pathlib.Path(k).stem: v
                    for k, v in ast.literal_eval(node.value).items()}
    raise RuntimeError("sync-common.py 里找不到 OP_FAMILIES（族 ↔ 节点单一事实源）")


# ---------- AST 工具 ----------

def sig_of(fn: ast.FunctionDef):
    """返回 (全部参数名, 必填参数名, 是否有 **kwargs)。"""
    a = fn.args
    params = [p.arg for p in a.posonlyargs + a.args + a.kwonlyargs]
    ndef = len(a.defaults) + len([d for d in a.kw_defaults if d is not None])
    return params, params[:len(params) - ndef], a.kwarg is not None


def _str_dict_keys(d: ast.Dict):
    """字面量字典的键集合；键非字符串常量（含 `{**x}`）时返回 None。"""
    keys: set[str] = set()
    for k in d.keys:
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            keys.add(k.value)
        else:
            return None
    return keys


def _module_const(module_tree: ast.Module, name: str):
    """模块级常量（老节点写法：`OP = "sd.checkpoint.load"`）。找不到返回 None。"""
    for node in module_tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == name for t in node.targets):
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                return node.value.value
    return None


def resolve_kwarg_keys(node: ast.AST, scope: ast.FunctionDef, tree: ast.Module):
    """把节点 main.py 的实参解析成键集合，解析不出来返回 None（调用方打 WARN）。

    支持的写法（都是仓里真实存在的形态）：
      run_op("op", {"image": ...})              → 字典字面量
      args = {"image": ...}; args["mask"] = ...  → 局部变量 + 条件追加（iclight 写法）
      submit_job(url, {"name": OP, "inputs": inputs})  → 老节点形态
    """
    if isinstance(node, ast.Dict):
        keys: set[str] = set()
        for k, v in zip(node.keys, node.values):
            if k is None:  # {**other}
                sub = resolve_kwarg_keys(v, scope, tree)
                if sub is None:
                    return None
                keys |= sub
            elif isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.add(k.value)
            else:
                return None
        return keys
    if isinstance(node, ast.Name):
        # 模块级常量（一般不是参数字典，但 `OP` 这种算子名常量走这里）
        const = _module_const(tree, node.id)
        if const is not None:
            return None
        for stmt in ast.walk(scope):
            if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == node.id for t in stmt.targets):
                keys = resolve_kwarg_keys(stmt.value, scope, tree)
                if keys is None:
                    return None
                for sub in ast.walk(scope):  # 条件追加 args["k"] = ...
                    if not (isinstance(sub, ast.Assign) and len(sub.targets) == 1):
                        continue
                    t = sub.targets[0]
                    if (isinstance(t, ast.Subscript)
                            and isinstance(t.value, ast.Name) and t.value.id == node.id):
                        if isinstance(t.slice, ast.Constant) \
                                and isinstance(t.slice.value, str):
                            keys.add(t.slice.value)
                        else:
                            return None  # 动态键 args[k] ⇒ 不猜
                return keys
        return None
    return None


# ---------- 解析服务端算子文件（族文件 / 遗留 server_op.py） ----------

def parse_ops(path: pathlib.Path) -> dict[str, dict]:
    """解析一个服务端算子文件 → {算子名: 契约}。

    两种注册形态都认（历史上并存，vton.py docstring 记过"写成平铺一行就注册不上"）：
      @registry.register("op", inputs={...})        # 装饰器
      registry.register("op", inputs={...})(fn)     # 手动应用
    """
    tree = ast.parse(path.read_text())
    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    ops: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, "attr", "") == "register"):
            continue
        # 过滤 _VersionedRegistry.register 内部的 f-string 转发调用
        if not node.args or not isinstance(node.args[0], ast.Constant) \
                or not isinstance(node.args[0].value, str) \
                or "." not in node.args[0].value:
            continue
        name = node.args[0].value
        kw = {k.arg: k.value for k in node.keywords}
        target = None
        for parent in ast.walk(tree):  # 手动应用：(…)(fn)
            if isinstance(parent, ast.Call) and parent.func is node:
                for a in parent.args:
                    if isinstance(a, ast.Name):
                        target = a.id
                        break
        for fn_def in fns.values():    # 装饰器位置
            if any(d is node for d in fn_def.decorator_list):
                target = fn_def.name
        info = {"inputs": kw.get("inputs") if isinstance(kw.get("inputs"), ast.Dict) else None,
                "outputs": kw.get("outputs") if isinstance(kw.get("outputs"), ast.Dict) else None,
                "fn": target, "lineno": node.lineno, "file": path}
        if target in fns:
            params, required, has_kwargs = sig_of(fns[target])
            info.update(params=params, required=required, has_kwargs=has_kwargs,
                        fn_node=fns[target])
        ops[name] = info
    return ops


def _return_dict_keys(fn: ast.FunctionDef):
    """函数所有 return 的键并集；只要有一个非字典字面量的 return 就返回 None（保守）。"""
    keys: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Return):
            if node.value is None or not isinstance(node.value, ast.Dict):
                return None
            sub = _str_dict_keys(node.value)
            if sub is None:
                return None
            keys |= sub
    return keys


# ---------- 检查：算子自身（E1/E2/E5） ----------

def check_op_contract(op: str, info: dict, label: str):
    errors: list[str] = []
    warns: list[str] = []
    rel = info["file"].relative_to(ROOT)
    where = f"{rel}:{info['lineno']}"
    if info["inputs"] is None:
        warns.append(f"{op}: inputs 不是字典字面量（{where}）⇒ 未静态校验")
    if info["fn"] is None or "params" not in info:
        errors.append(f"{label} E? {op}: 找不到注册目标函数（{where}；"
                      f"注册调用必须挂在函数上：装饰器位置或 (…)(fn)）")
        return errors, warns
    declared = _str_dict_keys(info["inputs"]) if info["inputs"] else set()
    if declared is not None:
        for port in sorted(declared - set(info["params"])):
            if not info["has_kwargs"]:
                errors.append(
                    f"{label} E1 {op}: 声明了 input '{port}' 但函数 {info['fn']}() 没有该"
                    f"参数（{where}）⇒ 引擎按声明名传 kwarg，真机 TypeError: "
                    f"unexpected keyword argument")
        for p in info["required"]:
            if p not in declared:
                errors.append(
                    f"{label} E2 {op}: 函数 {info['fn']}() 的必填参数 '{p}' 没进声明"
                    f"（{where}）⇒ 引擎永远不传，真机 TypeError: missing argument")
    if info["outputs"] is not None and "fn_node" in info:
        out_keys = _str_dict_keys(info["outputs"])
        ret_keys = _return_dict_keys(info["fn_node"])
        if out_keys is not None and ret_keys is not None:
            for port in sorted(out_keys - ret_keys):
                errors.append(
                    f"{label} E5 {op}: 声明了 output '{port}' 但函数 {info['fn']}() 的"
                    f"return 字典里没有（{where}）⇒ 运行时 did not return output '{port}'")
    return errors, warns


# ---------- 检查：节点调用点（E3/E4） ----------

def _calls_in(tree: ast.Module):
    """产出 (callee, call, fn_def) —— 只看 CALL_FORMS 里的调用。"""
    for fn_def in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        for call in ast.walk(fn_def):
            if not isinstance(call, ast.Call):
                continue
            callee = getattr(call.func, "id", None) or getattr(call.func, "attr", "")
            if callee in CALL_FORMS:
                yield callee, call, fn_def


def _call_op_and_args(callee: str, call: ast.Call, tree: ast.Module):
    """从调用里取出 (算子名, 参数字典节点)。取不到给 (None, None)。

    - `run_op("<op>", {…})` / `call_op(url, "<op>", {…})`：算子名是位置参数字符串
    - `submit_job(url, {"name": "<op>", "inputs": {…}})`：算子名在 payload 的 name 键
    """
    if callee == "submit_job":
        for a in call.args:
            if not isinstance(a, ast.Dict):
                continue
            kv = {}
            for k, v in zip(a.keys, a.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    kv[k.value] = v
            name_node, kw = kv.get("name"), kv.get("inputs")
            if name_node is None or kw is None:
                return None, None
            if isinstance(name_node, ast.Constant) and isinstance(name_node.value, str):
                return name_node.value, kw
            if isinstance(name_node, ast.Name):
                return _module_const(tree, name_node.id), kw
            return None, None
    op_name = None
    for a in call.args:
        if isinstance(a, ast.Constant) and isinstance(a.value, str) and "." in a.value:
            op_name = op_name or a.value
    if op_name is None:
        return None, None
    # 参数字典 = 算子名**后面**那个实参（call_op(url, "op", {…}) / run_op("op", {…})）
    arg_dict = None
    for i, a in enumerate(call.args):
        if isinstance(a, ast.Constant) and a.value == op_name:
            if i + 1 < len(call.args):
                arg_dict = call.args[i + 1]
            break
    if arg_dict is None:  # 关键字写法：run_op("op", inputs={…})
        for k in call.keywords:
            if k.arg == "inputs":
                arg_dict = k.value
    return op_name, arg_dict


def check_node_calls(node: str, path: pathlib.Path, index: dict):
    errors: list[str] = []
    warns: list[str] = []
    checked = 0
    tree = ast.parse(path.read_text())
    for callee, call, fn_def in _calls_in(tree):
        op_name, arg_node = _call_op_and_args(callee, call, tree)
        if op_name is None:
            warns.append(f"节点 {node}: {callee}() 的算子名不是字面量常量"
                         f"（{path.name}:{call.lineno}）⇒ 未静态校验")
            continue
        info = index.get(op_name)
        if info is None:
            warns.append(f"节点 {node}: 调用的 '{op_name}' 不在任何服务端算子文件里"
                         f"（{path.name}:{call.lineno}）⇒ 未静态校验")
            continue
        if arg_node is None:
            warns.append(f"节点 {node}: {op_name} 的实参不是可解析的字典"
                         f"（{path.name}:{call.lineno}）⇒ 未静态校验")
            continue
        keys = resolve_kwarg_keys(arg_node, fn_def, tree)
        if keys is None:
            warns.append(f"节点 {node}: {op_name} 的实参无法静态解析"
                         f"（{path.name}:{getattr(arg_node, 'lineno', call.lineno)}）"
                         f"⇒ 未静态校验")
            continue
        checked += 1
        declared = _str_dict_keys(info["inputs"]) if info["inputs"] else None
        if declared is None:
            continue
        for k in sorted(keys - declared):
            errors.append(
                f"[{node}] E3 传给 {op_name} 的 '{k}' 不在声明的 inputs 里"
                f"（{path.name}:{getattr(arg_node, 'lineno', call.lineno)}）⇒ "
                f"resolve_* 只遍历 op.inputs，该参数会被**静默丢弃**"
                f"（界面能填、流程 success、值没生效）")
        for p in info.get("required", []):
            if p not in keys:
                errors.append(
                    f"[{node}] E4 算子 {op_name} 的必填参数 '{p}' 没传给算子"
                    f"（{path.name}:{getattr(arg_node, 'lineno', call.lineno)}）⇒ "
                    f"真机 TypeError: missing argument")
    return errors, warns, checked


# ---------- 主体 ----------

def collect_sources():
    """返回 [(标签, 文件, 归属节点列表)]：两类服务端算子来源。"""
    out = []
    for fam, members in sorted(load_families().items()):
        out.append((f"族[{fam}]", ROOT / "_common" / "server_ops" / f"{fam}.py",
                    list(members)))
    managed = {m for _, _, ms in out for m in ms}
    for so in sorted(ROOT.glob("*/server_op.py")):
        node = so.parent.name
        if node not in managed:  # 遗留自带（未纳入 sync-common）
            out.append((f"遗留[{node}]", so, [node]))
    return out


def run(verbose=False):
    errors: list[str] = []
    warns: list[str] = []
    index: dict[str, dict] = {}
    n_files = 0
    for label, path, owners in collect_sources():
        if not path.is_file():
            errors.append(f"{label}: 服务端算子文件不存在 {path.relative_to(ROOT)}")
            continue
        n_files += 1
        ops = parse_ops(path)
        if not ops:
            errors.append(f"{label}: 解析不到任何 registry.register(\"族.算子\") 调用"
                          f"（{path.relative_to(ROOT)}；注册形态见 check-ops.py ⑥）")
        for op, info in sorted(ops.items()):
            if op in index:
                errors.append(
                    f"{label} E6 算子 '{op}' 已在 {index[op]['file'].relative_to(ROOT)} "
                    f"注册过 ⇒ 两个文件抢同一算子名，上传必 409（跨文件重名）")
                continue
            index[op] = {**info, "label": label, "owners": owners}
            e, w = check_op_contract(op, info, label)
            errors += e
            warns += w
            if verbose:
                print(f"  {label} {op}: 声明={sorted(_str_dict_keys(info['inputs']) or [])} "
                      f"签名={info.get('params')} 归属节点={owners}")

    n_nodes = n_calls = 0
    for mp in sorted(ROOT.glob("*/main.py")):
        if mp.parent.name.startswith("_"):
            continue
        n_nodes += 1
        e, w, checked = check_node_calls(mp.parent.name, mp, index)
        errors += e
        warns += w
        n_calls += checked

    for line in warns:
        print(f"WARN {line}")
    for line in errors:
        print(f"FAIL {line}")
    n_legacy = sum(1 for label, _, _ in collect_sources() if label.startswith("遗留"))
    # E5 只在"函数的 return 全是字典字面量"时可判定（其余返回变量/转发调用，不猜）
    e5_ok = sum(1 for info in index.values()
                if info["outputs"] is not None and "fn_node" in info
                and _return_dict_keys(info["fn_node"]) is not None)
    print(f"\n服务端算子契约一致性（gate O）：{n_files} 个算子文件"
          f"（族 {n_files - n_legacy} + 遗留 {n_legacy}）/ {len(index)} 算子 / "
          f"{n_nodes} 节点（静态校验 {n_calls} 个调用点；E5 可判定 {e5_ok}/{len(index)}），"
          f"FAIL {len(errors)} / WARN {len(warns)}")
    return 1 if errors else 0


# ---------- 负例自检（纪律：新增门禁必须带负例自检） ----------

# 模板用普通字符串（f-string 里的 {…} 转义太容易写错，且这里要嵌代码里的 {}）
_FAMILY_TMPL = '''
OP_VERSION = "deadbeefcafe"
OP_FAMILY = "selftest"


class _VersionedRegistry:
    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register("%s@%s" % (name, self._version), inputs,
                                    outputs, description)


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register("selftest.op", inputs={__INPUTS__},
                       outputs={"image": "IMAGE"}, description="stub")
    def _op(__SIG__):
        return __RET__
'''

_GOOD_FAMILY = _FAMILY_TMPL.replace(
    "__INPUTS__", '"image": "IMAGE", "scale": "FLOAT", "mode": "STRING"'
).replace("__SIG__", 'image, scale=2.0, mode="bilinear"').replace(
    "__RET__", '{"image": image, "info": f"{scale}{mode}"}')


def _family_with(inputs: str, fn_sig: str, ret: str = '{"image": image}') -> str:
    """按需拼一个族文件：inputs 声明 / 函数签名 / return 三者可独立改坏。"""
    return (_FAMILY_TMPL.replace("__INPUTS__", inputs)
            .replace("__SIG__", fn_sig).replace("__RET__", ret))


_LEGACY_TMPL = '''
def register(registry):
    @registry.register("legacy.op", inputs={__INPUTS__},
                       outputs={"image": "IMAGE"}, description="stub")
    def _op(__SIG__):
        return {"image": image}
'''


def _node_src(body: str) -> str:
    return ("from executor_base import run_op\n"
            "from flowx_client import param\n\n\n"
            "def main():\n" + body)


_GOOD_NODE = _node_src("    run_op('selftest.op', {'image': 'i', 'scale': 2.0, "
                       "'mode': 'cubic'}, emit_keys=['image'])\n")
# 老节点形态：submit_job(url, {"name": OP, "inputs": inputs})
_LEGACY_NODE = ("from flowx_client import param, submit_job\n\n"
                "OP = 'legacy.op'\n\n\n"
                "def main():\n"
                "    inputs = {'image': 'i'}\n"
                "    if param('x', ''):\n"
                "        inputs['scale'] = 3.0\n"
                "    submit_job('url', {'name': OP, 'inputs': inputs}, 'tok')\n")

_LEGACY_OP = _LEGACY_TMPL.replace("__INPUTS__", '"image": "IMAGE", "scale": "FLOAT"') \
    .replace("__SIG__", 'image, scale=3.0')

# 负例用例：(名字, 族源码 or None, {节点相对路径: 源码}, 期望命中的错误码 or None=期望全绿)
# ⚠️ 断言「具体错误码」而不是「rc≠0」——否则别的检查碰巧报错也会让负例假通过；
# ⚠️ 每个用例都自带**完整的**节点文件，保证命中的是预期那一类漂移。
NEG_CASES = [
    ("E1 声明了签名没有的端口",
     _family_with('"image": "IMAGE", "scale": "FLOAT"', "image, zoom=2.0"),
     {"selftest-node/main.py": _GOOD_NODE}, "E1 "),
    ("E2 必填参数没进声明",
     _family_with('"image": "IMAGE"', "image, scale"),
     {"selftest-node/main.py": _GOOD_NODE}, "E2 "),
    ("E5 声明 output 但 return 里没有",
     _family_with('"image": "IMAGE"', "image", ret='{"other": image}'),
     {"selftest-node/main.py": _GOOD_NODE}, "E5 "),
    ("E3 节点多传（会被静默丢弃）",
     _GOOD_FAMILY,
     {"selftest-node/main.py": _node_src(
         "    run_op('selftest.op', {'image': 'i', 'zoom': 2.0}, "
         "emit_keys=['image'])\n")}, "E3 "),
    ("E4 节点漏传必填",
     _family_with('"image": "IMAGE", "scale": "FLOAT"', "image, scale"),
     {"selftest-node/main.py": _node_src(
         "    run_op('selftest.op', {'image': 'i'}, emit_keys=['image'])\n")}, "E4 "),
    ("PASS 全对（装饰器形态）", _GOOD_FAMILY,
     {"selftest-node/main.py": _GOOD_NODE}, None),
    ("PASS 局部变量字典 + 条件追加（iclight 写法）",
     _family_with('"image": "IMAGE", "optional": "STRING"', "image, optional=''"),
     {"selftest-node/main.py": _node_src(
         "    args = {'image': 'i'}\n"
         "    if param('optional', ''):\n"
         "        args['optional'] = param('optional')\n"
         "    run_op('selftest.op', args, emit_keys=['image'])\n")}, None),
    ("WARN 动态键不 FAIL（不猜）", _GOOD_FAMILY,
     {"selftest-node/main.py": _node_src(
         "    args = {}\n    for k in ['image']:\n        args[k] = 'x'\n"
         "    run_op('selftest.op', args, emit_keys=['image'])\n")}, None),
    # 第二类来源：遗留自带 server_op.py（节点目录里，两个文件都带）
    ("E3 遗留算子：节点多传（submit_job + OP 常量）", None,
     {"selftest-node/main.py": _LEGACY_NODE.replace("inputs['scale'] = 3.0",
                                                    "inputs['zoom'] = 3.0"),
      "selftest-node/server_op.py": _LEGACY_OP}, "E3 "),
    ("PASS 遗留算子：submit_job + OP 常量 + 条件追加", None,
     {"selftest-node/main.py": _LEGACY_NODE,
      "selftest-node/server_op.py": _LEGACY_OP}, None),
    ("E6 两个文件抢同一算子名（上传必 409）", None,
     {"selftest-node/main.py": _LEGACY_NODE,
      "selftest-node/server_op.py": _LEGACY_OP,
      "other-node/main.py": _LEGACY_NODE,
      "other-node/server_op.py": _LEGACY_OP}, "E6 "),
]


def selftest():
    import contextlib
    import io
    import tempfile
    ok = total = 0
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td)
        (tmp / "_tools").mkdir()
        (tmp / "_common" / "server_ops").mkdir(parents=True)
        global ROOT
        real_root = ROOT
        for name, fam_src, files, want_code in NEG_CASES:
            # 每个用例重建最小仓：族文件（可无）+ 各节点目录的文件（显式声明）
            for old in (tmp / "_common" / "server_ops").glob("*.py"):
                old.unlink()
            for d in ("selftest-node", "other-node"):
                if (tmp / d).exists():
                    for f in (tmp / d).iterdir():
                        f.unlink()
            if fam_src is not None:
                (tmp / "_common" / "server_ops" / "selftest.py").write_text(fam_src)
            (tmp / "_tools" / "sync-common.py").write_text(
                'OP_FAMILIES = {"selftest.py": ["selftest-node"]}\n'
                if fam_src is not None else "OP_FAMILIES = {}\n")
            for rel, src in files.items():
                f = tmp / rel
                f.parent.mkdir(exist_ok=True)
                f.write_text(src)
            ROOT = tmp
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = run()
            out = buf.getvalue()
            hit = (rc == 0) if want_code is None else (want_code in out)
            total += 1
            ok += 1 if hit else 0
            codes = sorted({c for c in ("E1", "E2", "E3", "E4", "E5", "E6")
                            if f"{c} " in out})
            got = "全绿" if rc == 0 else ("+".join(codes) or "其它")
            detail = "\n".join(l for l in out.splitlines()
                               if l.startswith(("FAIL", "WARN")))[:200]
            print(f"  {'PASS' if hit else 'FAIL'} 负例 {name}: 期望 "
                  f"{'全绿' if want_code is None else want_code.strip()}，实得 {got}"
                  + (f"  [{detail}]" if detail and not hit else ""))
        ROOT = real_root
    print(f"\n门禁自检：{ok}/{total} 符合预期")
    return 0 if ok == total else 1


def main():
    if "--selftest" in sys.argv:
        return selftest()
    return run(verbose="--verbose" in sys.argv or "-v" in sys.argv)


if __name__ == "__main__":
    sys.exit(main())
