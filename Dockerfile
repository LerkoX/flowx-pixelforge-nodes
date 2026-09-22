# FlowX PixelForge 节点共享镜像（镜像节点，executor.bundled）
#
# 40 个节点结构一致（main.py + 共享 flowx_client.py，纯 HTTP 调推理服务），
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
COPY lora-loader/*.py        /opt/flowx-nodes/lora-loader/
COPY motion-loader/*.py      /opt/flowx-nodes/motion-loader/
COPY save-image/*.py         /opt/flowx-nodes/save-image/
COPY save-video/*.py         /opt/flowx-nodes/save-video/
COPY vae-decode/*.py         /opt/flowx-nodes/vae-decode/
COPY vae-encode/*.py         /opt/flowx-nodes/vae-encode/
COPY video-gen/*.py          /opt/flowx-nodes/video-gen/

COPY sd3-clip-encode/*.py    /opt/flowx-nodes/sd3-clip-encode/
COPY sd3-empty-latent/*.py   /opt/flowx-nodes/sd3-empty-latent/
COPY sd3-sampler/*.py        /opt/flowx-nodes/sd3-sampler/
COPY sd3-vae-decode/*.py     /opt/flowx-nodes/sd3-vae-decode/
COPY sd3-vae-encode/*.py     /opt/flowx-nodes/sd3-vae-encode/

# --- 推理服务客户端（纯 HTTP；A 类，2026-09-22 补齐） ---
COPY clip-vision-load/*.py           /opt/flowx-nodes/clip-vision-load/
COPY cond-average/*.py               /opt/flowx-nodes/cond-average/
COPY cond-combine/*.py               /opt/flowx-nodes/cond-combine/
COPY cond-set-area/*.py              /opt/flowx-nodes/cond-set-area/
COPY controlnet-apply/*.py           /opt/flowx-nodes/controlnet-apply/
COPY controlnet-load/*.py            /opt/flowx-nodes/controlnet-load/
COPY detail-refine/*.py              /opt/flowx-nodes/detail-refine/
COPY embedding-load/*.py             /opt/flowx-nodes/embedding-load/
COPY image-upscale/*.py              /opt/flowx-nodes/image-upscale/
COPY ipadapter-apply/*.py            /opt/flowx-nodes/ipadapter-apply/
COPY ipadapter-load/*.py             /opt/flowx-nodes/ipadapter-load/
COPY latent-composite/*.py           /opt/flowx-nodes/latent-composite/
COPY latent-set-noise-mask/*.py      /opt/flowx-nodes/latent-set-noise-mask/
COPY latent-upscale/*.py             /opt/flowx-nodes/latent-upscale/
COPY preprocess-canny/*.py           /opt/flowx-nodes/preprocess-canny/
COPY preprocess-openpose/*.py        /opt/flowx-nodes/preprocess-openpose/
COPY upscale-model-apply/*.py        /opt/flowx-nodes/upscale-model-apply/
COPY upscale-model-load/*.py         /opt/flowx-nodes/upscale-model-load/
COPY vae-decode-video/*.py           /opt/flowx-nodes/vae-decode-video/
COPY vae-load/*.py                   /opt/flowx-nodes/vae-load/
COPY video-sample-latent/*.py        /opt/flowx-nodes/video-sample-latent/

# 冒烟自检：所有节点目录必须存在且含 main.py
RUN set -e; for d in /opt/flowx-nodes/*/; do test -f "$d/main.py" || { echo "missing main.py in $d"; exit 1; }; done
