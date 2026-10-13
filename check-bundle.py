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
  K. widget 声明 kind:'file'（本机上传控件）的节点必须 local-only
     （上传物落在 FlowX 机器，docker/远端执行器看不到该磁盘）
  L. widget 交互回归：node 里用 DOM stub 真跑一遍 bundle
     （上传控件写回参数 / 绑定态与回放态只读 / 未声明就不出上传入口 /
     放大弹层的 ✕・点空白・Esc 三条关闭途径；1.3.0 事故：fileControl 漏传
     onClose 且 Esc 监听挂在 mediaImageParam 分支 ⇒ 弹窗关不掉）

用法：python3 nodes/check-bundle.py [--quiet]
退出码：0 一致；1 有不一致（逐条打印人话）。
"""
from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys

# 仅本地运行的节点：代码仍在仓库里（save-image/save-video 的 .py 也进了镜像，
# 但执行器必须 local；图像/蒙版节点因镜像内无 Pillow 连分发也不做）
# 执行器纪律（2026-10-13 用户拍板）：**只有"读/写本机文件"的节点才允许 local-only**，
# 其余一律要支持 docker。本集合即"确证碰本机文件系统"的节点，声明里出现 docker/image/bundled
# 会被 gate E 拦下（image-batch-load 读本机目录、image-batch-save 写本机文件）。
LOCAL_ONLY = {"save-image", "save-video", "load-image", "mask-paint",
              "image-batch-load", "image-batch-save"}

# 已删除的 Studio 业务端点（去业务耦合：4 个业务端点 → 2 个通用代理）：
# widget 里再引用就是指向 404 的死链接，第三方 API 路径语义必须自持。
# 注释行（说明历史/迁移背景）不参与判定。
DEAD_ENDPOINTS = ("/inference/models-files", "/input-image",
                  "/op-replay", "/interrupt-inference")


def check_flowx_json_schema(root: pathlib.Path):
    """gate P：flowx.json **结构**校验（导入 API 的硬要求，手写节点最容易错）。

    为什么需要（2026-10-13 实测）：手写新节点时把 `mock`/`ui` 写成了字符串路径，
    Studio `POST /nodes/import` 直接 400（`cannot unmarshal string into Go struct
    field NodePackage.mock`）——这类错误本可以在提交前拦下。校验：
      必填 name/version/entry/parameters/outputs/executor；
      entry/mock.entry/ui.entry 为**对象**且指向真实存在的文件；
      parameters[].name 唯一；executor.supportedTypes 非空。
    """
    errs, warns = [], []
    for fj in sorted(root.glob("*/flowx.json")):
        node = fj.parent.name
        try:
            d = json.loads(fj.read_text())
        except Exception as e:  # noqa: BLE001
            errs.append(f"{node}: flowx.json 不是合法 JSON（{e}）")
            continue
        # 硬要求字段（导入 API 会 400 或生态硬依赖；实测 `language` 缺 ⇒ 400）
        for key in ("name", "displayName", "description", "version", "author", "icon",
                    "tags", "language", "entry", "files", "ui", "executor",
                    "parameters", "outputs"):
            if key not in d:
                errs.append(f"{node}: 缺字段 {key}（导入 API / 生态约定要求）")
        # 约定字段：缺省不报错（默认行为可用），但提示 —— 便于新节点与存量对齐
        for key in ("extract", "timeout", "env", "mock"):
            if key not in d and (fj.parent / "mock.py").is_file() if key == "mock" else key not in d:
                warns.append(f"{node}: 未声明 {key}（走默认；mask-paint/flux-mask-refine 等"
                             f"历史节点同样缺省，属已知口径）")
        if not isinstance(d.get("executor"), dict) or not d["executor"].get("supportedTypes"):
            errs.append(f"{node}: executor.supportedTypes 缺失或为空")
        # entry / mock / ui 必须是结构体（不是字符串路径）
        if not isinstance(d.get("entry"), str) or not (fj.parent / str(d.get("entry"))).is_file():
            errs.append(f"{node}: entry={d.get('entry')!r} 不存在")
        # mock/ui 在 Go 结构里是 omitempty（可缺省）；**存在时**必须是对象且文件在
        mk = d.get("mock")
        if mk is not None and not isinstance(mk, dict):
            errs.append(f"{node}: mock 存在但不是对象 {{enabled,entry}}（导入 API 要求；"
                        f"got {type(mk).__name__}）")
        elif isinstance(mk, dict) and mk.get("entry") \
                and not (fj.parent / mk["entry"]).is_file():
            errs.append(f"{node}: mock.entry={mk['entry']!r} 文件不存在")

        ui = d.get("ui")
        if ui is not None and not isinstance(ui, dict):
            errs.append(f"{node}: ui 存在但不是对象 {{entry,width,height}}"
                        f"（got {type(ui).__name__}）")
        elif isinstance(ui, dict) and ui.get("entry") \
                and not (fj.parent / ui["entry"]).is_file():
            errs.append(f"{node}: ui.entry={ui['entry']!r} 文件不存在")
        names = [p.get("name") for p in d.get("parameters") or []]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            errs.append(f"{node}: parameters 名字重复 {sorted(dup)}")
        if any(not n for n in names):
            errs.append(f"{node}: parameters 有空 name")
    return errs, warns


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

    # J. 算子族 ↔ 成员节点 main.py 一致（跨文件重名必 409 / 注册表被切坏）
    # gate M：节点参数引用一致性（param() / env 模板 / 声明未使用）——
    # 真机事故教训：参数改名后 flowx.json 与 widget 都改了、main.py 忘了改，
    # 本地单测与产物校验都发现不了，直到真机 TypeError 才暴露。
    r = subprocess.run([sys.executable, str(root / "_tools" / "check-node-params.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        problems.extend((r.stdout + r.stderr).strip().splitlines() or
                        ["节点参数引用一致性失败（gate M）"])
    r = subprocess.run([sys.executable, str(root / "_tools" / "check-ops.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        problems.extend(line for line in
                        (r.stdout + r.stderr).strip().splitlines() if line.strip())

    # O. 服务端算子「声明 ↔ 函数签名 ↔ 节点实参」一致性（gate O，覆盖族文件 15 个 +
    # 遗留自带 server_op.py 24 个）。目标是最阴的一类漂移：节点传了声明里没有的键 ⇒
    # resolve_* 只遍历 op.inputs ⇒ 参数被**静默丢弃**（界面能填、流程 success、值没生效）；
    # 以及声明/签名不对齐导致的真机 TypeError（只到真机才暴露）。
    r = subprocess.run([sys.executable, str(root / "_tools" / "check-op-signatures.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        problems.extend(line for line in
                        (r.stdout + r.stderr).strip().splitlines()
                        if line.startswith("FAIL"))

    # N. main() 桩冒烟（gate N）——`node mock` 跑的是 mock.py，跑不到 main.py；
    # 2026-10 连续两批真机事故（S2 的 NameError、S3 的可选参数）都是"只在 main.py
    # 里、只有真机才暴露"。门禁用桩把每个节点 main() 真跑一遍。
    r = subprocess.run([sys.executable, str(root / "_tools" / "test-main-smoke.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        problems.extend(line for line in
                        (r.stdout + r.stderr).strip().splitlines()
                        if line.startswith("FAIL"))

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
                      or "emit_result_preview" in src or has_model_field)
        if not needs_host:
            continue
        params = {p["name"] for p in
                  json.loads((nd / "flowx.json").read_text())["parameters"]}
        if "service_url_host" not in params:
            problems.append(
                f"{nd.name}: 有画布侧读取（模型下拉/预览帧）但 flowx.json 未声明 "
                f"service_url_host 参数（docker 节点需另给 Studio 可达地址）")
        reports_preview = ("emit_preview" in src
                            or "emit_result_preview" in src)
        if reports_preview and "host_base" not in src and "emit_result_preview" not in src:
            problems.append(
                f"{nd.name}: 直接上报预览帧必须经 flowx_client.host_base() 构造 "
                f"Studio 宿主机可达的帧 URL")

    # K. 上传控件（widget kind:'file'）与笔画外置存储（paint.strokesRefKey）只能声明在
    #    local-only 节点。上传物与笔画 JSON 都落在 FlowX 机器（Studio 的 media.roots，
    #    默认 ~/flowx-input/uploads/），路径参数由节点读盘；docker/远端执行器看不到
    #    这台机器的磁盘 ⇒ 声明了就是“能画能传但一跑就 FileNotFound”的坑。
    #    需要上传/外置存储语义的节点必须 supportedTypes: [local]。
    for wd in sorted(root.glob("*/ui/widget-def.js")):
        src = wd.read_text()
        declares = []
        if "kind: 'file'" in src:
            declares.append("kind:'file'（本机上传）")
        if "strokesRefKey" in src:
            declares.append("paint.strokesRefKey（笔画 JSON 外置到本机文件）")
        if not declares:
            continue
        nd = wd.parent.parent
        fj = nd / "flowx.json"
        if not fj.is_file():
            continue
        ex = json.loads(fj.read_text()).get("executor") or {}
        types = ex.get("supportedTypes") or []
        if "docker" in types:
            problems.append(
                f"{nd.name}: widget 声明了{'、'.join(declares)}但 executor."
                f"supportedTypes={types} 含 docker/远端 —— 上传物/笔画文件在本机，"
                f"容器里读不到；请改为 supportedTypes: [local]")

    # Q. pipeline/*.yaml 的节点引用与参数声明（引擎展开阶段是硬校验：
    #    "config.params references undeclared parameter" 会让整跑在展开期失败）
    r = subprocess.run([sys.executable, str(root / "_tools" / "check-pipeline.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        problems.extend([ln for ln in (r.stdout + r.stderr).splitlines()
                         if ln.strip().startswith("-")] or
                        ["pipeline 校验失败（跑 python3 nodes/_tools/check-pipeline.py 看详情）"])

    # P. flowx.json 结构（导入 API 的硬要求：entry/mock/ui 形状 + 文件存在）
    schema_errs, schema_warns = check_flowx_json_schema(root)
    problems.extend(schema_errs)
    schema_warns = [] if quiet else schema_warns

    # L. widget 交互回归（DOM stub 里真跑 bundle；产物字节合法 ≠ 点得动）
    node_bin = shutil.which("node")
    if node_bin:
        r = subprocess.run([node_bin, str(root / "_tools" / "test-widget-ui.mjs")],
                           capture_output=True, text=True)
        if r.returncode != 0:
            fails = [ln for ln in (r.stdout + r.stderr).splitlines() if ln.startswith("FAIL")]
            problems.extend(fails or ["widget UI 回归失败（跑 node nodes/_tools/test-widget-ui.mjs 看详情）"])
    elif not quiet:
        print("! 未找到 node，跳过 widget UI 回归（gate L）", file=sys.stderr)

    for w in schema_warns:
        print(f"! {w}", file=sys.stderr)
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
