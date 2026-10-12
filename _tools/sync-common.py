#!/usr/bin/env python3
"""sync-common.py：把 _common/ 下的共享件分发到各节点目录（单一事实源）。

为什么需要：`flowx_client.py` 曾分裂成 3 个变体（有无 ensure_plugin、emit_preview
新旧签名），改协议字段时要逐目录改、极易漂移。现在唯一事实源放在 `_common/`，
各节点目录里的副本由本脚本生成；`check-bundle.py` 会校验"副本 == _common 版本"，
漂移即发布失败。

用法：
  python3 _tools/sync-common.py          # 分发（写各节点目录）
  python3 _tools/sync-common.py --check  # 只校验，不改文件（check-bundle 会调用）
"""
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ---------- F1 族版本戳（dev-plan §47）----------
# 算子身份 = 实现内容：族文件注册名统一加 "@<版本>" 后缀，版本 = 本文件内容 hash 前 12 位
# （计算时把 stamp 行归零，避免自引用；幂等，重复执行结果相同）。
# 效果：改过实现的族换名换文件 ⇒ 新旧版本在服务端并存、互不顶替；未改动的族跨版本同名。
STAMP_RE = re.compile(r'^OP_VERSION\s*=\s*"[0-9a-f]*"', re.M)
STAMP_ZERO = 'OP_VERSION = "000000000000"'


def family_version(text: str) -> str:
    """由族文件内容算版本（stamp 行归零后取 sha256 前 12 位）。"""
    probe = STAMP_RE.sub(STAMP_ZERO, text, count=1)
    return hashlib.sha256(probe.encode("utf-8")).hexdigest()[:12]


def stamp_family(text: str, fam: str):
    """写回族版本戳。返回 (新文本, 版本)；缺 stamp 行则报错（F1 合同）。"""
    if not STAMP_RE.search(text):
        raise SystemExit(
            f"✗ 族文件 {fam} 缺 `OP_VERSION = \"...\"` 行（F1 合同，见 dev-plan §47）")
    ver = family_version(text)
    return STAMP_RE.sub(f'OP_VERSION = "{ver}"', text, count=1), ver

# 共享件 → 分发到的节点目录判定
def targets(name, node_dir):
    main = node_dir / "main.py"
    body = main.read_text() if main.exists() else ""
    if name == "flowx_client.py":
        return True                      # 所有节点都调推理服务/emit
    if name == "executor_base.py":
        return "executor_base" in body   # 只有声明用它的节点需要
    if name == "image_filters.py":
        return "image_filters" in body   # 只有 G5 滤镜族节点需要
    if name == "image_batch.py":
        return "image_batch" in body     # 批量 I/O 族（B5/52.2）
    if name == "image_montage.py":
        return "image_montage" in body   # 拼图节点（B5/52.2）
    return False


# 算子族文件（_common/server_ops/<族>.py）→ 成员节点的 server_op.py。
# 同族多算子共享 helper，单文件上传机制下同族同文件可避免跨文件重名 409；
# 集中放 _common 由本脚本分发，防多副本漂移（同 flowx_client 教训）。
OP_FAMILIES = {
    "cond_latent.py": ["cond-average", "cond-combine", "cond-set-area",
                       "latent-composite", "latent-set-noise-mask",
                       "latent-upscale"],
    # 三算子族（canny/depth/openpose）：与历史服务端已上传的 preprocess_canny.py
    # 同内容 ⇒ hash 命中不重传
    "preprocess.py": ["preprocess-canny", "preprocess-depth",
                      "preprocess-openpose"],
    # G7 预处理器族（7 算子）：独立文件 ⇒ 上传名 preprocess_<op>.py 不与三算子族撞车
    "preprocess_g7.py": ["preprocess-anyline", "preprocess-lineart",
                         "preprocess-softedge", "preprocess-scribble",
                         "preprocess-mlsd", "preprocess-normal",
                         "preprocess-shuffle"],
    "ipadapter.py": ["ipadapter-load", "ipadapter-apply", "clip-vision-load"],
    "flux.py": ["flux-clip-encode", "flux-detail-refine", "flux-dual-clip-load",
                "flux-empty-latent", "flux-sampler", "flux-unet-load",
                "flux-vae-decode", "flux-vae-encode", "flux-vae-load"],
    "instantid.py": ["instantid-apply", "instantid-face-analyze",
                     "face-mask", "face-similarity"],
    "upscale.py": ["upscale-model-load", "upscale-model-apply"],
    # G1 抠图族（B2，dev-plan §48）：BiRefNet 抠图（变换沿 refine/composite 走节点侧 PIL）
    "matting.py": ["matting-birefnet"],
    # G6 人脸修复族（B2，dev-plan §48）：spandrel GFPGAN/RestoreFormer + insightface 检测
    "face_restore.py": ["face-restore"],
    # G2 分割族（B3，dev-plan §49）：SAM1 点/框/自动分割 + GroundingDINO 文本检测
    "seg.py": ["sam-load", "sam-segment", "sam-auto-segment",
               "grounding-dino-load", "grounding-detect"],
    # G4 分块重绘超分族（B3，dev-plan §49）：切片规划（纯计算）+ 逐块重绘与接缝融合
    "tiled.py": ["tiled-upscale-plan", "tiled-upscale-apply"],
    # G8 上色族（B4，dev-plan §50）：DDColor（装载预热 + 上色推理）
    "colorize.py": ["colorize-load", "colorize"],
    # G9 重打光族（B4，dev-plan §50）：IC-Light（装载预热 + 重打光推理）
    "iclight.py": ["iclight-load", "iclight"],
    "video_sample.py": ["video-gen", "video-sample-latent"],
}


def main():
    check = "--check" in sys.argv
    drift, wrote = [], 0
    sources = sorted(p for p in (ROOT / "_common").glob("*.py"))
    nodes = sorted(p for p in ROOT.iterdir()
                   if p.is_dir() and (p / "flowx.json").exists())
    for src in sources:
        want = src.read_text()
        for node in nodes:
            if not targets(src.name, node):
                continue
            dst = node / src.name
            if not dst.exists() or dst.read_text() != want:
                if check:
                    drift.append(str(dst.relative_to(ROOT)))
                else:
                    dst.write_text(want)
                    wrote += 1
    # 算子族分发：_common/server_ops/<族>.py → 成员节点 server_op.py
    # 分发前先按内容打版本戳（F1：算子名 = <基名>@<族内容 hash12>）
    fam_dir = ROOT / "_common" / "server_ops"
    versions = {}
    for fam in sorted(OP_FAMILIES):
        src = fam_dir / fam
        text = src.read_text()
        stamped, ver = stamp_family(text, fam)
        versions[fam] = ver
        if stamped != text:
            if check:
                drift.append(f"_common/server_ops/{fam}（版本戳过期"
                             f"：应为 {ver}）")
            else:
                src.write_text(stamped)
                wrote += 1
    for fam, members in sorted(OP_FAMILIES.items()):
        src = fam_dir / fam
        want = src.read_text()
        for member in members:
            dst = ROOT / member / "server_op.py"
            if not dst.exists() or dst.read_text() != want:
                if check:
                    drift.append(str(dst.relative_to(ROOT)))
                else:
                    dst.write_text(want)
                    wrote += 1
    if check:
        if drift:
            print(f"✗ 共享件漂移（{len(drift)} 个）：{drift[:5]}"
                  f"{' ...' if len(drift) > 5 else ''}")
            print("  修复：python3 nodes/_tools/sync-common.py")
            return 1
        print("✓ 共享件一致（_common → 各节点目录）")
        return 0
    print(f"✓ 已同步共享件：写入 {wrote} 个文件（源：{', '.join(p.name for p in sources)}"
          f" + server_ops 族文件 {len(OP_FAMILIES)} 个）")
    print("  族版本戳：" + ", ".join(f"{k}@{v}" for k, v in sorted(versions.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
