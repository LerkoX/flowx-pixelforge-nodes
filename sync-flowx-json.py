#!/usr/bin/env python3
"""把 BUNDLE_VERSION 同步到所有节点 flowx.json 的 image 字段（executor.bundled 一并置真）。

LOCAL_ONLY 中的节点仅本地运行，不同步镜像字段（其代码仍随镜像分发，仅作备份，
不会被 docker 引用；图像/蒙版节点因镜像内无 Pillow 连分发也不做）。
"""
import json
import pathlib
import sys

LOCAL_ONLY = {"save-image", "save-video",
              # 第一档纯客户端图像/蒙版节点：PIL 处理在 Studio 侧，仅 local 执行器
              "image-rotate", "image-flip", "image-crop", "image-composite",
              "mask-from-image", "mask-to-image", "mask-grow", "mask-feather",
              "mask-invert"}

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
