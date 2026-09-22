#!/usr/bin/env python3
"""节点镜像清单一致性检查（发布门槛）。

单一事实源：`Dockerfile` 的 COPY 行 = "该节点代码真的在镜像里"。
本脚本把该清单与各节点 flowx.json 的声明对表，任何一边漂移都非 0 退出：

  A. 声明 supportedTypes 含 docker ⇒ 必须在 COPY 清单里
     （否则 docker 执行时必挂 `cd: /opt/flowx-nodes/<name>: No such file`，exec 363/364 事故）
  B. 声明 docker ⇒ 必须有 image 且等于 BUNDLE_VERSION 对应的 tag
     （docker 容器靠这个 tag 起，缺了会跑到无名镜像上）
  C. 写了 image 或 bundled=true ⇒ 必须在 COPY 清单里（不许"假装在镜像里"）
  D. 在 COPY 清单里且不属于 LOCAL_ONLY ⇒ image 必须等于当前 tag 且 bundled=true
  E. LOCAL_ONLY 节点 ⇒ 声明里不许出现 docker/image/bundled（本地专用，不伪装可 docker）

用法：python3 nodes/check-bundle.py [--quiet]
退出码：0 一致；1 有不一致（逐条打印人话）。
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

# 仅本地运行的节点：代码仍在仓库里（save-image/save-video 的 .py 也进了镜像，
# 但执行器必须 local；图像/蒙版节点因镜像内无 Pillow 连分发也不做）
LOCAL_ONLY = {"save-image", "save-video", "load-image"}


def in_image_dirs(root: pathlib.Path) -> dict[str, str]:
    """解析 Dockerfile 的 COPY <dir>/*.py → {节点目录: 目标目录}"""
    out: dict[str, str] = {}
    for line in (root / "Dockerfile").read_text().splitlines():
        m = re.match(r"^COPY\s+([\w\-.]+)/\*\.py\s+(\S+)/?\s*$", line.strip())
        if m:
            out[m.group(1)] = m.group(2)
    return out


def main() -> int:
    quiet = "--quiet" in sys.argv
    root = pathlib.Path(__file__).parent
    version = (root / "BUNDLE_VERSION").read_text().strip()
    tag = f"lerkobba/flowx-pixelforge-nodes:v{version}"
    copied = in_image_dirs(root)

    problems: list[str] = []
    checked = 0
    for fj in sorted(root.glob("*/flowx.json")):
        d = json.loads(fj.read_text())
        name = d["name"]
        checked += 1
        ex = d.get("executor") or {}
        types = ex.get("supportedTypes") or []
        declares_docker = "docker" in types
        image = d.get("image")
        bundled = bool(ex.get("bundled"))
        in_image = name in copied

        if declares_docker and not in_image:
            problems.append(
                f"{name}: 声明了 docker 但不在镜像 COPY 清单里"
                f"（docker 执行会挂 No such file；请补 Dockerfile COPY 或去掉 docker）"
            )
        if declares_docker and image != tag:
            problems.append(f"{name}: 声明了 docker 但 image={image!r}，应为 {tag}")
        if (image or bundled) and not in_image:
            problems.append(
                f"{name}: image/bundled 声明了「在镜像里」但不在 COPY 清单里"
                f"（image={image!r} bundled={bundled}）"
            )
        if name in LOCAL_ONLY:
            if declares_docker or image or bundled:
                problems.append(
                    f"{name}: 本地专用节点不得声明 docker/image/bundled"
                    f"（types={types} image={image!r} bundled={bundled}）"
                )
        elif in_image:
            if image != tag:
                problems.append(f"{name}: 在镜像里但 image={image!r}，应为 {tag}（跑 sync-flowx-json.py 同步）")
            if not bundled:
                problems.append(f"{name}: 在镜像里但未声明 executor.bundled=true")

    if problems:
        print(f"✗ 清单不一致（{len(problems)} 处）：", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 1

    if not quiet:
        print(f"✓ 清单一致：{checked} 个节点包 / 镜像内 {len(copied)} 个目录 / tag {tag}")
        print(f"  docker 可用: {sum(1 for fj in root.glob('*/flowx.json') if 'docker' in (json.loads(fj.read_text()).get('executor') or {}).get('supportedTypes', []))}"
              f" 个；local-only: {sum(1 for fj in root.glob('*/flowx.json') if json.loads(fj.read_text())['name'] in LOCAL_ONLY)} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
