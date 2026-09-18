#!/usr/bin/env bash
# 构建并推送 pixelforge 节点共享镜像。
# 任何节点代码变更后：bump BUNDLE_VERSION → 跑本脚本 → 同步各 flowx.json 的 image tag
# （可用同目录 sync-flowx-json.py 一键同步）。
set -euo pipefail
cd "$(dirname "$0")"

TAG="lerkobba/flowx-pixelforge-nodes:v$(tr -d '[:space:]' < BUNDLE_VERSION)"
echo "==> building $TAG"
docker build -t "$TAG" .
echo "==> pushing $TAG"
docker push "$TAG"
echo "==> done: $TAG"
