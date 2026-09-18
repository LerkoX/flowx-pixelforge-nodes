#!/usr/bin/env python3
"""把 BUNDLE_VERSION 同步到所有节点 flowx.json 的 image 字段（executor.bundled 一并置真）。

LOCAL_ONLY 中的节点（save-image/save-video）仅本地运行：保存文件须落在 Studio
宿主机媒体白名单目录，不同步镜像字段（其代码仍随镜像分发，仅作备份，不会被 docker 引用）。
"""
import json
import pathlib
import sys

LOCAL_ONLY = {"save-image", "save-video"}

root = pathlib.Path(__file__).parent
tag = "lerkobba/flowx-pixelforge-nodes:v" + (root / "BUNDLE_VERSION").read_text().strip()

for fj in sorted(root.glob("*/flowx.json")):
    d = json.loads(fj.read_text())
    if d["name"] in LOCAL_ONLY:
        print(f"{d['name']:<18} skipped (local-only)")
        continue
    d["image"] = tag
    ex = d.setdefault("executor", {})
    ex["bundled"] = True
    fj.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
    print(f"{d['name']:<18} image={tag} bundled=true")
print("synced", file=sys.stderr)
