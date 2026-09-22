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
  F. 各节点目录里的共享件（flowx_client.py / executor_base.py）必须与 _common/ 一致
     （唯一事实源；曾分裂成 3 个变体导致改协议要逐目录改）
  G. 各节点 ui/node-widget.js 必须等于 base + widget-def 重新拼接的结果
     （防"改了 _common/widget-base.js 忘了重建"，产物是随包分发的）
  H. widget 不得引用已删除的 Studio 业务端点（去业务耦合后只留通用代理；
     洏网过：模型下拉残留 /inference/models-files → 404，流水线 76 事故）
  I. 有画布侧读取的节点（模型下拉、预览帧上报）必须声明 service_url_host，
     且预览 URL 必须经 host_base() 构造（Studio 宿主机可达 ≠ 容器内可达；
     洏网过：docker 节点 model 下拉/预览拽 172.17.0.1:8100 → Studio 502）

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

# 已删除的 Studio 业务端点（去业务耦合：4 个业务端点 → 2 个通用代理）：
# widget 里再引用就是指向 404 的死链接，第三方 API 路径语义必须自持。
# 注释行（说明历史/迁移背景）不参与判定。
DEAD_ENDPOINTS = ("/inference/models-files", "/input-image",
                  "/op-replay", "/interrupt-inference")


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

    # F. 共享件漂移（_common/ 是唯一事实源）
    import subprocess
    r = subprocess.run([sys.executable, str(root / "_tools" / "sync-common.py"), "--check"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        problems.append(r.stdout.strip() or "共享件漂移")

    # G. widget 产物漂移（base + widget-def 重建对表）
    r = subprocess.run([sys.executable, str(root / "_tools" / "build-widget.py"),
                        "--all", "--check"], capture_output=True, text=True)
    if r.returncode != 0:
        problems.extend(line for line in r.stderr.strip().splitlines() if line.strip())

    # H. widget 引用已删除的 Studio 业务端点
    for wj in sorted(root.glob("*/ui/node-widget.js")):
        for i, line in enumerate(wj.read_text().splitlines(), 1):
            s = line.strip()
            if s.startswith("*") or s.startswith("//") or s.startswith("/*"):
                continue
            for ep in DEAD_ENDPOINTS:
                if ep in s:
                    problems.append(
                        f"{wj.parent.parent.name}: ui/node-widget.js:{i} 引用已删除的 "
                        f"Studio 端点 {ep}（改用 /api/v1/service-proxy）")

    # I. 画布侧读取地址声明（模型下拉/预览帧都由 Studio 发起 HTTP，
    #    docker 节点的 service_url 是容器内网地址，Studio 取不到）
    for main_py in sorted(root.glob("*/main.py")):
        nd = main_py.parent
        src = main_py.read_text()
        wd = nd / "ui" / "widget-def.js"
        has_model_field = wd.is_file() and "kind: 'model'" in wd.read_text()
        needs_host = ("executor_base" in src or "emit_preview" in src
                      or has_model_field)
        if not needs_host:
            continue
        params = {p["name"] for p in
                  json.loads((nd / "flowx.json").read_text())["parameters"]}
        if "service_url_host" not in params:
            problems.append(
                f"{nd.name}: 有画布侧读取（模型下拉/预览帧）但 flowx.json 未声明 "
                f"service_url_host 参数（docker 节点需另给 Studio 可达地址）")
        if "emit_preview" in src and "host_base" not in src:
            problems.append(
                f"{nd.name}: 直接上报预览帧必须经 flowx_client.host_base() 构造 "
                f"Studio 宿主机可达的帧 URL")

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
