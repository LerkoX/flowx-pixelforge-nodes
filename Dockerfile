# FlowX PixelForge 节点共享镜像（镜像节点，executor.bundled）
#
# 各节点结构一致（main.py + 共享 flowx_client.py / image_io.py / image_filters.py），
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
COPY instantid-face-analyze/*.py     /opt/flowx-nodes/instantid-face-analyze/
COPY instantid-apply/*.py            /opt/flowx-nodes/instantid-apply/
COPY flux-unet-load/*.py             /opt/flowx-nodes/flux-unet-load/
COPY flux-dual-clip-load/*.py        /opt/flowx-nodes/flux-dual-clip-load/
COPY flux-vae-load/*.py              /opt/flowx-nodes/flux-vae-load/
COPY flux-empty-latent/*.py          /opt/flowx-nodes/flux-empty-latent/
COPY flux-clip-encode/*.py           /opt/flowx-nodes/flux-clip-encode/
COPY flux-sampler/*.py               /opt/flowx-nodes/flux-sampler/
COPY flux-redux-apply/*.py           /opt/flowx-nodes/flux-redux-apply/
COPY flux-vae-decode/*.py            /opt/flowx-nodes/flux-vae-decode/
COPY flux-vae-encode/*.py            /opt/flowx-nodes/flux-vae-encode/
COPY flux-detail-refine/*.py         /opt/flowx-nodes/flux-detail-refine/
COPY flux-mask-refine/*.py         /opt/flowx-nodes/flux-mask-refine/
COPY latent-composite/*.py           /opt/flowx-nodes/latent-composite/
COPY latent-set-noise-mask/*.py      /opt/flowx-nodes/latent-set-noise-mask/
COPY latent-upscale/*.py             /opt/flowx-nodes/latent-upscale/
COPY model-unload/*.py               /opt/flowx-nodes/model-unload/
COPY preprocess-canny/*.py           /opt/flowx-nodes/preprocess-canny/
COPY preprocess-openpose/*.py        /opt/flowx-nodes/preprocess-openpose/
COPY preprocess-depth/*.py          /opt/flowx-nodes/preprocess-depth/
COPY preprocess-anyline/*.py         /opt/flowx-nodes/preprocess-anyline/
COPY preprocess-lineart/*.py         /opt/flowx-nodes/preprocess-lineart/
COPY preprocess-mlsd/*.py            /opt/flowx-nodes/preprocess-mlsd/
COPY preprocess-normal/*.py          /opt/flowx-nodes/preprocess-normal/
COPY preprocess-scribble/*.py        /opt/flowx-nodes/preprocess-scribble/
COPY preprocess-shuffle/*.py         /opt/flowx-nodes/preprocess-shuffle/
COPY preprocess-softedge/*.py        /opt/flowx-nodes/preprocess-softedge/
COPY face-mask/*.py                 /opt/flowx-nodes/face-mask/
COPY face-similarity/*.py           /opt/flowx-nodes/face-similarity/
COPY upscale-model-apply/*.py        /opt/flowx-nodes/upscale-model-apply/
COPY upscale-model-load/*.py         /opt/flowx-nodes/upscale-model-load/
COPY vae-decode-video/*.py           /opt/flowx-nodes/vae-decode-video/
COPY vae-load/*.py                   /opt/flowx-nodes/vae-load/
COPY video-sample-latent/*.py        /opt/flowx-nodes/video-sample-latent/
COPY vram-ops/*.py                /opt/flowx-nodes/vram-ops/

# --- Studio 侧图像/蒙版（PIL；图片本体经服务对象仓库传递，不在本地文件系统上） ---
COPY image-composite/*.py                /opt/flowx-nodes/image-composite/
COPY image-crop/*.py                     /opt/flowx-nodes/image-crop/
COPY image-flip/*.py                     /opt/flowx-nodes/image-flip/
COPY image-rotate/*.py                   /opt/flowx-nodes/image-rotate/
COPY mask-feather/*.py                   /opt/flowx-nodes/mask-feather/
COPY mask-from-image/*.py                /opt/flowx-nodes/mask-from-image/
COPY mask-grow/*.py                      /opt/flowx-nodes/mask-grow/
COPY mask-invert/*.py                    /opt/flowx-nodes/mask-invert/
COPY mask-to-image/*.py                  /opt/flowx-nodes/mask-to-image/

# --- G5 传统滤镜族（节点侧纯 PIL；共享实现 _common/image_filters.py） ---
COPY image-adjust/*.py                     /opt/flowx-nodes/image-adjust/
COPY image-channel-extract/*.py            /opt/flowx-nodes/image-channel-extract/
COPY image-channel-merge/*.py              /opt/flowx-nodes/image-channel-merge/
COPY image-blur/*.py                       /opt/flowx-nodes/image-blur/
COPY image-curves/*.py                     /opt/flowx-nodes/image-curves/
COPY image-hsl/*.py                        /opt/flowx-nodes/image-hsl/
COPY image-noise/*.py                      /opt/flowx-nodes/image-noise/
COPY image-sharpen/*.py                    /opt/flowx-nodes/image-sharpen/

# B2 G1 抠图族（dev-plan §四十八）：BiRefNet 抠图（走推理服务算子）+ alpha 精修 +
# 换底合成（后两个纯节点侧 PIL）
COPY matting-birefnet/*.py                 /opt/flowx-nodes/matting-birefnet/
COPY matting-refine/*.py                   /opt/flowx-nodes/matting-refine/
COPY matting-composite/*.py                /opt/flowx-nodes/matting-composite/

# B2 G6 人脸修复族（dev-plan §四十八）：insightface 检测 + spandrel 修复
COPY face-restore/*.py                     /opt/flowx-nodes/face-restore/
# B3（dev-plan §49）：SAM 分割 5 节点 + 分块超分 2 节点（均为服务端算子族的成员节点）
COPY sam-load/*.py                         /opt/flowx-nodes/sam-load/
COPY sam-segment/*.py                      /opt/flowx-nodes/sam-segment/
COPY sam-auto-segment/*.py                 /opt/flowx-nodes/sam-auto-segment/
COPY grounding-dino-load/*.py               /opt/flowx-nodes/grounding-dino-load/
COPY grounding-detect/*.py                  /opt/flowx-nodes/grounding-detect/
COPY tiled-upscale-plan/*.py                /opt/flowx-nodes/tiled-upscale-plan/
COPY tiled-upscale-apply/*.py               /opt/flowx-nodes/tiled-upscale-apply/
COPY colorize-load/*.py                   /opt/flowx-nodes/colorize-load/
COPY colorize/*.py                        /opt/flowx-nodes/colorize/
COPY iclight-load/*.py                    /opt/flowx-nodes/iclight-load/
COPY iclight/*.py                         /opt/flowx-nodes/iclight/
COPY image-watermark/*.py                 /opt/flowx-nodes/image-watermark/
# 文本渲染字体随节点包进镜像（local/docker 逐字节一致；gate 只解析 *.py 行故不计入目录数）
COPY image-watermark/fonts/                /opt/flowx-nodes/image-watermark/fonts/

COPY image-shadow/*.py                    /opt/flowx-nodes/image-shadow/
COPY catvton-load/*.py                      /opt/flowx-nodes/catvton-load/
COPY catvton-tryon/*.py                     /opt/flowx-nodes/catvton-tryon/
COPY image-montage/*.py                     /opt/flowx-nodes/image-montage/
COPY image-pad/*.py                       /opt/flowx-nodes/image-pad/
# Pillow 供图像/蒙版/滤镜族节点使用（image-* / mask-* 等）。远端 docker 宿主无外网，
# 故把 manylinux cp311 wheel 随上下文带进去离线安装（_wheels/，见 README）。
COPY _wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels Pillow==12.3.0 \
 && rm -rf /tmp/wheels

# 冒烟自检：所有节点目录必须存在且含 main.py
RUN set -e; for d in /opt/flowx-nodes/*/; do test -f "$d/main.py" || { echo "missing main.py in $d"; exit 1; }; done
