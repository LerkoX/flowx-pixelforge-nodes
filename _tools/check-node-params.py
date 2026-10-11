#!/usr/bin/env python3
"""check-node-params.py（gate M）：节点参数引用一致性。

为什么需要：真机事故——`matting-refine` 的参数从 `min_area` 改名 `despeckle` 后，
`flowx.json` 与 widget/test 都改了，唯独 `main.py` 仍 `param("min_area")`，
本地单测（只测 process()）与 check-bundle（只看产物）都发现不了，
直到真机 e2e 报 `TypeError: process() got an unexpected keyword argument` 才暴露。

校验项：
  A. `main.py` 里每个 `param("<name>")` / `param("<name>", ...)` 的 `<name>`
     必须在 `flowx.json` 的 `parameters[].name` 里（或等于已知保留名）。
  B. 反过来：`flowx.json` 里声明的参数若在 `main.py` 里从没被读取，给出**警告**
     （允许：仅由上游绑定/UI 使用的展示类参数——用 `--warn-only` 语义，不影响退出码）。
  C. `env` 段引用的 `{{ Param.<name> }}` 必须存在于 parameters。
  D. `mock.py` 里的 `param("<name>")` 同样校验（mock 用于参数注入回归）。

用法：python3 nodes/_tools/check-node-params.py [节点目录名...]   （退出码 0=通过）
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 由运行环境注入、不需要在 flowx.json 里声明的保留参数
RESERVED = {"service_url", "service_url_host", "service_token"}

PARAM_RE = re.compile(r'param\(\s*"([^"]+)"')
# 真机事故：param(..., cast=bool) 对 env 注入的字符串 "false" 会得到 True
# （bool("false") 为真）⇒ 布尔参数必须走 image_io.pbool()
CAST_BOOL_RE = re.compile(r'param\(\s*"[^"]+"\s*,[^)]*cast\s*=\s*bool')
ENV_RE = re.compile(r"\{\{\s*Param\.([A-Za-z0-9_]+)\s*\}\}")


def main() -> int:
    only = [a for a in sys.argv[1:] if not a.startswith("-")]
    problems: list[str] = []
    warns: list[str] = []
    nodes = sorted(p for p in ROOT.iterdir()
                   if p.is_dir() and (p / "flowx.json").exists())
    checked = 0
    for node in nodes:
        if only and node.name not in only:
            continue
        try:
            cfg = json.loads((node / "flowx.json").read_text())
        except Exception as e:
            problems.append(f"{node.name}: flowx.json 解析失败：{e}")
            continue
        declared = {p["name"] for p in cfg.get("parameters", [])}
        env = cfg.get("env", {}) or {}

        # C. env 引用
        for k, v in env.items():
            for name in ENV_RE.findall(str(v)):
                if name not in declared:
                    problems.append(
                        f"{node.name}: env.{k} 引用未声明的参数 Param.{name}")

        # A. main.py / mock.py 的 param() 引用
        used: set[str] = set()
        for fname in ("main.py", "mock.py"):
            f = node / fname
            if not f.exists():
                continue
            body = f.read_text()
            if CAST_BOOL_RE.search(body):
                problems.append(
                    f"{node.name}/{fname}: 用了 param(..., cast=bool) —— env 注入的是"
                    f"字符串，bool(\"false\") 为真 ⇒ 布尔参数必须用 image_io.pbool()")
            for name in PARAM_RE.findall(body):
                used.add(name)
                if name not in declared and name not in RESERVED:
                    problems.append(
                        f"{node.name}/{fname}: param(\"{name}\") 未在 flowx.json "
                        f"parameters 里声明（改名后忘了同步 main.py？）")
        # B. 声明未使用（警告）
        for name in sorted(declared - used - RESERVED):
            if f"{{{{ Param.{name} }}}}" in json.dumps(env):
                continue          # 由 env 直传也算被使用
            warns.append(f"{node.name}: 参数 {name} 已声明但 main.py/mock.py 未读取")
        checked += 1

    for w in warns:
        print(f"! {w}")
    if problems:
        print(f"✗ 参数引用一致性问题（{len(problems)} 条）：")
        for p in problems:
            print(f"  - {p}")
        print("  修复：让 flowx.json 的 parameters / env 与 main.py/mock.py 对齐")
        return 1
    print(f"✓ 参数引用一致：{checked} 个节点（param() 引用、env 模板、声明未使用警告）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
