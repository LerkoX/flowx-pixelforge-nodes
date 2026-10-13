#!/usr/bin/env python3
"""校验 `pipeline/*.yaml`：每个节点的 nodeRef 与 `params` 必须与仓库节点声明一致。

为什么需要（2026-10-13 实测）：模板给 `flux-clip-encode` 传了它**没声明**的
`service_url_host`，Studio 引擎在展开阶段直接硬失败：
`failed to expand node ...: config.params references undeclared parameter`——
这类错误本可以在本地提交前拦下（一条 GPU 流水线跑到一半才炸，代价是几十分钟）。

校验项：
  1. `nodeRef: <name>@<version>` 在仓库里存在（版本必须精确匹配某个 flowx.json）；
  2. `params` 的每个键都在该节点的 parameters 声明里（引擎是硬校验）；
  3. 必填参数（required: true）已被提供（可来自 params 或上游连线，故只告警）；
  4. `executor` 与节点 supportedTypes 相容（声明 local-only 的不能写 docker）。

用法：python3 nodes/_tools/check-pipeline.py [pipeline/xxx.yaml …]
"""
import json
import pathlib
import sys

try:
    import yaml
except ImportError:  # pragma: no cover
    print("需要 PyYAML", file=sys.stderr)
    sys.exit(2)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_nodes():
    out = {}
    for fj in sorted(ROOT.glob("*/flowx.json")):
        d = json.loads(fj.read_text())
        out[(d["name"], d["version"])] = d
    return out


def check_file(path, nodes, problems, warns):
    try:
        doc = yaml.safe_load(path.read_text())
    except Exception as e:  # noqa: BLE001
        problems.append(f"{path.name}: YAML 解析失败 {e}")
        return 0
    if not isinstance(doc, dict) or "Nodes" not in doc:
        problems.append(f"{path.name}: 缺 Nodes 段")
        return 0
    n_checked = 0
    for node_id, spec in (doc.get("Nodes") or {}).items():
        cfg = (spec or {}).get("config") or {}
        ref = cfg.get("nodeRef") or ""
        if "@" not in ref:
            problems.append(f"{path.name}/{node_id}: nodeRef 缺版本 {ref!r}")
            continue
        name, ver = ref.rsplit("@", 1)
        decl = nodes.get((name, ver))
        if decl is None:
            avail = sorted(v for (n, v) in nodes if n == name)
            problems.append(f"{path.name}/{node_id}: {ref} 在仓库不存在"
                            f"（该节点仓库版本：{avail or '无此节点名'}）")
            continue
        n_checked += 1
        declared = {p["name"] for p in decl.get("parameters") or []}
        for k in (cfg.get("params") or {}):
            if k not in declared:
                problems.append(f"{path.name}/{node_id}: {ref} 未声明参数 {k!r}"
                                f"（引擎展开阶段硬失败；已声明：{sorted(declared)}）")
        types = (decl.get("executor") or {}).get("supportedTypes") or []
        ex = cfg.get("executor")
        if ex and types and ex not in types:
            problems.append(f"{path.name}/{node_id}: executor={ex} 不在 {ref} 的 "
                            f"supportedTypes={types}")
        provided = set(cfg.get("params") or {}) | set(cfg.get("inputs") or {})
        for p in decl.get("parameters") or []:
            if p.get("required") and p["name"] not in provided:
                warns.append(f"{path.name}/{node_id}: {ref} 必填参数 {p['name']!r} 未在 "
                             f"params 里给出（可能靠上游连线，或确有遗漏）")
    return n_checked


def main():
    nodes = load_nodes()
    targets = [pathlib.Path(a) for a in sys.argv[1:]] or sorted((ROOT.parent / "pipeline").glob("*.yaml"))
    problems, warns, total = [], [], 0
    for p in targets:
        total += check_file(p, nodes, problems, warns)
    print(f"校验 {len(targets)} 个流水线 / {total} 个节点实例 / 仓库节点 {len(nodes)} 个版本")
    for w in warns:
        print(f"! {w}")
    if problems:
        print(f"✗ {len(problems)} 处不一致：")
        for x in problems:
            print("  - " + x)
        return 1
    print("✓ 节点引用 / 参数声明 / 执行器相容性全部一致")
    return 0


if __name__ == "__main__":
    sys.exit(main())
