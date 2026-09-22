#!/usr/bin/env python3
"""按镜像清单把 BUNDLE_VERSION / bundled 同步到各节点 flowx.json。

单一事实源是 `Dockerfile` 的 COPY 行（=节点代码真的在镜像里）：
  - 在清单里且非 LOCAL_ONLY  → image=<当前 tag>、executor.bundled=true
  - 在清单外（或 LOCAL_ONLY）→ 清掉 image 与 bundled（不许"假装在镜像里"）

`check-bundle.py` 反向校验声明与清单一致（发布门槛）。两个脚本配合，
本轮之前那种"flowx.json 写了 image、镜像里却没这个目录"的漂移不会再出现。
"""
import json
import pathlib
import re
import sys

# 仅本地运行的节点：即使 .py 进了镜像也不得声明 image/bundled（执行器必须 local）
LOCAL_ONLY = {"save-image", "save-video", "load-image",
              # 纯客户端图像/蒙版节点：PIL 处理在 Studio 侧（镜像内无 Pillow，也不分发）
              "image-rotate", "image-flip", "image-crop", "image-composite",
              "mask-from-image", "mask-to-image", "mask-grow", "mask-feather",
              "mask-invert"}

root = pathlib.Path(__file__).parent
tag = "lerkobba/flowx-pixelforge-nodes:v" + (root / "BUNDLE_VERSION").read_text().strip()
copied = {m.group(1) for line in (root / "Dockerfile").read_text().splitlines()
          if (m := re.match(r"^COPY\s+([\w\-.]+)/\*\.py\s+\S+/\s*$", line.strip()))}

changed = 0
for fj in sorted(root.glob("*/flowx.json")):
    d = json.loads(fj.read_text())
    name = d["name"]
    ex = d.setdefault("executor", {})
    before = (d.get("image"), ex.get("bundled"))
    if name in copied and name not in LOCAL_ONLY:
        d["image"] = tag
        ex["bundled"] = True
        note = f"image={tag} bundled=true"
    else:
        d.pop("image", None)
        ex.pop("bundled", None)
        note = "cleared (not in image / local-only)"
    after = (d.get("image"), ex.get("bundled"))
    if before != after:
        changed += 1
    fj.write_text(json.dumps(d, ensure_ascii=False, indent=2) + "\n")
    print(f"{name:<24} {note}")
print(f"\n{changed} 个文件有变化；tag={tag}，镜像内 {len(copied)} 个目录", file=sys.stderr)
