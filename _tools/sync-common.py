#!/usr/bin/env python3
"""sync-common.py：把 _common/ 下的共享件分发到各节点目录（单一事实源）。

为什么需要：`flowx_client.py` 曾分裂成 3 个变体（有无 ensure_plugin、emit_preview
新旧签名），改协议字段时要逐目录改、极易漂移。现在唯一事实源放在 `_common/`，
各节点目录里的副本由本脚本生成；`check-bundle.py` 会校验"副本 == _common 版本"，
漂移即发布失败。

用法：
  python3 _tools/sync-common.py          # 分发（写各节点目录）
  python3 _tools/sync-common.py --check  # 只校验，不改文件（check-bundle 会调用）
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 共享件 → 分发到的节点目录判定
def targets(name, node_dir):
    main = node_dir / "main.py"
    body = main.read_text() if main.exists() else ""
    if name == "flowx_client.py":
        return True                      # 所有节点都调推理服务/emit
    if name == "executor_base.py":
        return "executor_base" in body   # 只有声明用它的节点需要
    return False


def main():
    check = "--check" in sys.argv
    drift, wrote = [], 0
    sources = sorted(p for p in (ROOT / "_common").glob("*.py"))
    nodes = sorted(p for p in ROOT.iterdir()
                   if p.is_dir() and (p / "flowx.json").exists())
    for src in sources:
        want = src.read_text()
        for node in nodes:
            if not targets(src.name, node):
                continue
            dst = node / src.name
            if not dst.exists() or dst.read_text() != want:
                if check:
                    drift.append(str(dst.relative_to(ROOT)))
                else:
                    dst.write_text(want)
                    wrote += 1
    if check:
        if drift:
            print(f"✗ 共享件漂移（{len(drift)} 个）：{drift[:5]}"
                  f"{' ...' if len(drift) > 5 else ''}")
            print("  修复：python3 nodes/_tools/sync-common.py")
            return 1
        print("✓ 共享件一致（_common → 各节点目录）")
        return 0
    print(f"✓ 已同步共享件：写入 {wrote} 个文件（源：{', '.join(p.name for p in sources)}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
