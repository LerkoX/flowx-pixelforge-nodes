#!/usr/bin/env python3
"""算子族 ↔ 节点 main.py 一致性静态校验（发布门槛，check-bundle 调用）。

为什么需要：服务端插件上传历史上踩过两次：
  1) 10 个预处理器共用一族文件 → 重传时撞上既有 `preprocess_canny.py` 的三个算子（409）；
  2) 族文件 register() 被脚本切坏（注册表只剩 op 名字符串）→ 上传成功但注册 0 个算子。
外加 F1（dev-plan §47）算子身份版本化后新增三类事故：
  3) 族文件忘了跑 sync-common（版本戳 ≠ 内容 hash）→ 改了实现却不换名，旧版会顶掉新版；
  4) OP_FAMILY 写错 → 落盘名与族不对应，同族节点各传各的文件（又回到 409/歧义）；
  5) register() 里漏写 `_VersionedRegistry(registry, OP_VERSION)` 包装 → 注册的还是不带版本
     的旧名，F1 归零。

本脚本静态比对（不 import、不需要 controlnet_aux/torch，纯文本）：
  A. 每个 `_common/server_ops/<族>.py` 里 `registry.register("<op>")` 的算子集合
  B. `_tools/sync-common.py` OP_FAMILIES 每个成员节点 `main.py` 的 `run_op("<op>")`
断言：①成员节点要跑的算子必须在同族文件里注册；②族文件里注册的算子必须有成员节点在用；
③同一算子不得出现在两个族文件里；④**族文件版本戳 == 内容 hash 且 OP_FAMILY == 文件 stem**；
⑤**register() 内的注册调用都在版本化包装之后**。

用法：python3 nodes/_tools/check-ops.py   （退出码 0 一致 / 1 有问题）
"""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REG_RE = re.compile(r'registry\.register\(\s*[\'"]([\w\.]+)[\'"]')
OP_FAMILY_RE = re.compile(r'^OP_FAMILY\s*=\s*["\']([\w\.]+)["\']', re.M)
WRAP_RE = re.compile(r'_VersionedRegistry\(registry,\s*OP_VERSION\)')
# ⑥ 注册调用的返回值必须被用掉（装饰器位置 或 手动 )(fn) 应用），见 check_register_calls
import ast
# 节点侧声明算子的三种写法（executor_base.run_op / 自包含节点的
# ensure_plugin + call_op）：取并集，正常每节点恰好 1 个算子
OP_PATTERNS = (
    re.compile(r'run_op\(\s*[\'"]([\w\.]+)[\'"]'),
    re.compile(r'call_op\(\s*[\w\.]+,\s*[\'"]([\w\.]+)[\'"]'),
    re.compile(r'ensure_plugin\(\s*[\w\.]+,\s*[\'"]([\w\.]+)[\'"]'),
)


def _load_sync_common():
    """导入 sync-common.py（复用其 family_version/stamp 口径，单一事实源）。"""
    path = ROOT / "_tools" / "sync-common.py"
    spec = importlib.util.spec_from_file_location("flowx_sync_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_families() -> dict[str, list[str]]:
    """从 sync-common.py 读 OP_FAMILIES（唯一事实源，避免两处维护）。"""
    src = (ROOT / "_tools" / "sync-common.py").read_text()
    ns: dict = {}
    start = src.index("OP_FAMILIES = {")
    end = src.index("\n}\n", start) + 3
    exec(compile(src[start:end], "sync-common.py", "exec"), ns)
    return ns["OP_FAMILIES"]


def check_register_calls(fam: str, src: str) -> list[str]:
    """⑥ 的判定体：register() 内的注册调用必须把返回值用掉。

    合法两种写法：
      A. 装饰器：  @registry.register("op", ...)  挂在算子函数上
      B. 手动应用：registry.register("op", ...)(fn)
    非法：平铺一行 `registry.register(...)`（返回值被丢弃 ⇒ 注册表里零算子）。
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as e:  # 语法错误由其它门禁兜（这里不重复报）
        return [f"{fam}: 语法错误 {e}"]
    reg_fn = next((n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "register"), None)
    if reg_fn is None:
        return []
    calls = [n for n in ast.walk(reg_fn)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "register"]
    used = set()
    for n in ast.walk(reg_fn):
        # A. 装饰器位置
        for d in getattr(n, "decorator_list", []) or []:
            if isinstance(d, ast.Call):
                used.add(id(d))
        # B. 手动应用：register(...)(fn) —— 外层 Call 的 func 是注册调用
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Call):
            used.add(id(n.func))
    out = []
    for c in calls:
        if id(c) not in used:
            out.append(f"{fam}:{c.lineno}: 注册调用未挂函数（返回值被丢弃）——"
                       f"必须写成装饰器或手动 )(fn) 应用，否则服务端注册表零算子"
                       f"（vton 族 exec 707 事故）")
    return out


def main() -> int:
    problems: list[str] = []
    families = load_families()
    sync = _load_sync_common()
    fam_ops: dict[str, set[str]] = {}
    op_owner: dict[str, str] = {}
    for fam, members in families.items():
        path = ROOT / "_common" / "server_ops" / fam
        if not path.is_file():
            problems.append(f"{fam}: 族文件不存在（{path}）")
            continue
        src = path.read_text()
        ops = set(REG_RE.findall(src))
        fam_ops[fam] = ops
        if not ops:
            problems.append(f"{fam}: register() 里一个算子都没注册"
                            f"（注册表被切坏/漏写）")
        # ---- F1 版本化校验（dev-plan §47）----
        m = OP_FAMILY_RE.search(src)
        if not m:
            problems.append(f"{fam}: 缺 `OP_FAMILY = \"...\"`（F1 合同）")
        elif m.group(1) != path.stem:
            problems.append(f"{fam}: OP_FAMILY={m.group(1)!r} 须等于文件 stem "
                            f"{path.stem!r}（否则同族节点会各传各的文件）")
        expect = sync.family_version(src)
        if f'OP_VERSION = "{expect}"' not in src:
            problems.append(f"{fam}: 版本戳 ≠ 内容 hash（应为 {expect}）"
                            f"——请跑 python3 nodes/_tools/sync-common.py")
        fdef = src.find("def register(registry)")
        wrap = WRAP_RE.search(src, fdef if fdef >= 0 else 0)
        first_reg = src.find("registry.register(", fdef if fdef >= 0 else 0)
        if fdef < 0:
            problems.append(f"{fam}: 找不到 def register(registry)")
        elif wrap is None or (first_reg >= 0 and wrap.start() > first_reg):
            problems.append(
                f"{fam}: register() 里没有在注册之前包装 "
                f"`registry = _VersionedRegistry(registry, OP_VERSION)`"
                f"（F1：否则注册的还是不带版本的旧名）")
        # ⑥ 注册调用的返回值必须被用掉（vton 族首版真机事故，exec 707）：
        #    Registry.register() 返回**装饰器**，只有被应用到函数上才会写进注册表。
        #    平铺调用（返回值丢弃）的后果：上传 200 + 归属簿记账（claimed 来自录制
        #    shim）+ 其它静态门禁全过，但服务端注册表零算子 ⇒ 运行时 "unknown op"。
        problems += check_register_calls(fam, src)
        for op in ops:
            if op in op_owner:
                problems.append(
                    f"算子 {op} 同时被 {op_owner[op]} 与 {fam} 注册"
                    f"（跨文件重名 ⇒ 上传必 409：请合并到同一族文件）")
            op_owner[op] = fam
        used: set[str] = set()
        for member in members:
            mp = ROOT / member / "main.py"
            if not mp.is_file():
                problems.append(f"{fam}: 成员节点 {member} 缺 main.py")
                continue
            body = mp.read_text()
            calls = sorted({m for pat in OP_PATTERNS for m in pat.findall(body)})
            if not calls:
                # 条件选算子（如 video-gen: op_name = "video.sample_latent"
                # if output_mode == "latent" else "video.sample"）
                m = re.search(r'op_name\s*=\s*(.+)', body)
                if m:
                    # 只认带点的算子名，避免把 "latent"/"video" 这类模式串算进来
                    calls = sorted(set(re.findall(
                        r'[\'"]([\w]+\.[\w\.]+)[\'"]', m.group(1))))
            if not calls:
                problems.append(f"{member}/main.py: 认不出声明的算子（三种写法都没有）")
                continue
            used.update(calls)
            for op in calls:
                if op not in ops:
                    problems.append(
                        f"{member}/main.py 跑算子 {op}，但族文件 {fam} 未注册它"
                        f"（运行期会报「插件实际注册了 …」）")
            if len(calls) > 1 and member != "video-gen":
                problems.append(f"{member}/main.py: 声明算子 {calls}（多于 1 个，需确认）")
        for orphan in sorted(ops - used):
            problems.append(f"{fam}: 算子 {orphan} 没有成员节点在用（孤儿算子）")
        if len(members) != len(ops):
            problems.append(f"{fam}: 成员节点 {len(members)} 个 vs 注册算子 "
                            f"{len(ops)} 个（一对一关系被破坏）")

    if problems:
        print(f"✗ 算子族一致性检查失败（{len(problems)} 条）：")
        for p in problems:
            print("  -", p)
        return 1
    total = sum(len(v) for v in fam_ops.values())
    print(f"✓ 算子族一致：{len(fam_ops)} 个族文件 / {total} 个算子 / "
          f"{sum(len(m) for m in families.values())} 个成员节点，无跨文件重名")
    return 0


if __name__ == "__main__":
    sys.exit(main())
