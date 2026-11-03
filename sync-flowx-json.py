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
# 仅本地运行的节点（与 check-bundle.py 的同名集合必须一致：
# 改了分类两处一起改，否则 sync 会写 image/bundled、check 又判它违规）
LOCAL_ONLY = {"save-image", "save-video", "load-image", "load-video", "mask-paint",
              "image-batch-load", "image-batch-save", "save-mesh"}

root = pathlib.Path(__file__).parent
tag = "lerkobba/flowx-pixelforge-nodes:v" + (root / "BUNDLE_VERSION").read_text().strip()
_dockerfile = (root / "Dockerfile").read_text()
copied = {m.group(1) for line in _dockerfile.splitlines()
          if (m := re.match(r"^COPY\s+([\w\-.]+)/\*\.py\s+\S+/\s*$", line.strip()))}
if not copied and re.search(r"^COPY\s+\.\s+\S+", _dockerfile, re.M):
    # Dockerfile 已改为【全量单层 COPY】（71f1c2e：绕过经典 builder 127 层上限）：
    # 镜像内目录 = 所有含 flowx.json 的目录（.dockerignore 已排除 ui/ mock.py 等）。
    # 不识别这种写法时 copied 恒为空 ⇒ 所有节点都被误判成"不在镜像里"而清掉
    # image/bundled（2026-11 实测踩中：sync 把 118 个 flowx.json 全清了，
    # 与 check-bundle.py 的清单解析口径分裂）。两边的判定必须永远同源。
    copied = {d.name for d in root.iterdir()
              if d.is_dir() and (d / "flowx.json").is_file()}

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
