# FlowX PixelForge 节点共享镜像（镜像节点，executor.bundled）
#
# 各节点结构一致（main.py + 共享 flowx_client.py / image_io.py / image_filters.py），
# 共用一个镜像：每个节点的代码放在 /opt/flowx-nodes/<name>/，
# Studio 展开器按节点名 cd 到对应目录直接运行，不从 Studio 拉取任何文件。
# ui/ 资产（浏览器加载）与 mock.py（本地沙箱用）不进镜像。
#
# 构建/推送：build-image.sh（bump BUNDLE_VERSION 后跑）
FROM python:3.11-slim

LABEL org.opencontainers.image.title="flowx-pixelforge-nodes" \
      org.opencontainers.image.description="FlowX PixelForge bundled node image (all nodes share one image)"

WORKDIR /opt/flowx-nodes
# ---------------------------------------------------------------------------
# 一条 COPY 带走全部节点目录。
#
# 为什么不逐节点 `COPY <name>/*.py ...`：Docker **经典 builder 有 127 层上限**，
# 126 个节点逐条 COPY 会直接超限（v1.20.0 实测 `max depth exceeded`）。
# .dockerignore 已把 `*/ui/`、`*/mock.py`、`_tools/`、`_widget-template/`、
# `__pycache__/`、`*.pyc` 排除在上下文外 —— 即"浏览器侧/本地沙箱/门禁脚本不进镜像"
# 的口径不变；下面再用两条 RUN 把非节点残留清干净。
#
# 与 check-bundle.py 的关系：清单门禁识别 `COPY . `（全量），镜像内目录 = 所有含
# flowx.json 的目录 ⇒ "声明 docker 但镜像里没有"这类事故从根上不可能再发生。
# ---------------------------------------------------------------------------
COPY . /opt/flowx-nodes/

# Pillow 供图像/蒙版/滤镜族节点使用（image-* / mask-* 等）。远端 docker 宿主无外网，
# 故把 manylinux cp311 wheel 随上下文带进镜像离线安装（_wheels/，见 README）。
RUN pip install --no-cache-dir --no-index --find-links=/opt/flowx-nodes/_wheels Pillow==12.3.0

# 清掉上下文里的非运行期残留：顶层文件（Dockerfile/README/LICENSE/...）、
# 仓库元数据、wheel 缓存，以及任何不含 main.py 的目录（_common/_tools/...）
RUN set -e; cd /opt/flowx-nodes; \
    find . -maxdepth 1 -type f -delete; \
    rm -rf .github .dockerignore .gitignore _wheels _tools _widget-template _common; \
    for d in */; do [ -f "$d/main.py" ] || rm -rf "$d"; done

# 冒烟自检：所有节点目录必须存在且含 main.py
RUN set -e; for d in /opt/flowx-nodes/*/; do test -f "$d/main.py" || { echo "missing main.py in $d"; exit 1; }; done
