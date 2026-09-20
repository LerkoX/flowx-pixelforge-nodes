# FlowX PixelForge 节点共享镜像（镜像节点，executor.bundled）
#
# 14 个节点结构一致（main.py + 共享 flowx_client.py，纯 HTTP 调推理服务），
# 共用一个镜像：每个节点的代码放在 /opt/flowx-nodes/<name>/，
# Studio 展开器按节点名 cd 到对应目录直接运行，不从 Studio 拉取任何文件。
# ui/ 资产（浏览器加载）与 mock.py（本地沙箱用）不进镜像。
#
# 构建/推送：nodes/build-image.sh（bump BUNDLE_VERSION 后跑）
FROM python:3.11-slim

LABEL org.opencontainers.image.title="flowx-pixelforge-nodes" \
      org.opencontainers.image.description="FlowX PixelForge bundled node image (all nodes share one image)"

WORKDIR /opt/flowx-nodes

COPY checkpoint-loader/*.py  /opt/flowx-nodes/checkpoint-loader/
COPY clip-text-encode/*.py   /opt/flowx-nodes/clip-text-encode/
COPY empty-latent/*.py       /opt/flowx-nodes/empty-latent/
COPY inference-ensure/*.py   /opt/flowx-nodes/inference-ensure/
COPY inference-op/*.py       /opt/flowx-nodes/inference-op/
COPY ksampler/*.py           /opt/flowx-nodes/ksampler/
COPY load-image/*.py         /opt/flowx-nodes/load-image/
COPY lora-loader/*.py        /opt/flowx-nodes/lora-loader/
COPY motion-loader/*.py      /opt/flowx-nodes/motion-loader/
COPY save-image/*.py         /opt/flowx-nodes/save-image/
COPY save-video/*.py         /opt/flowx-nodes/save-video/
COPY sd3-txt2img/*.py        /opt/flowx-nodes/sd3-txt2img/
COPY vae-decode/*.py         /opt/flowx-nodes/vae-decode/
COPY vae-encode/*.py         /opt/flowx-nodes/vae-encode/
COPY video-gen/*.py          /opt/flowx-nodes/video-gen/

# 冒烟自检：所有节点目录必须存在且含 main.py
RUN set -e; for d in /opt/flowx-nodes/*/; do test -f "$d/main.py" || { echo "missing main.py in $d"; exit 1; }; done
