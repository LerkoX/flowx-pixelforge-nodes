"""人脸修复算子族（G6 / B2 批次，dev-plan §四十八）。

- face.restore：IMAGE → IMAGE（检测人脸 → 裁剪 → 修复 → 羽化贴回）

模型（spandrel 加载，权重放 FACERESTORE_MODELS_DIR，默认 /models/facerestore）：
- gfpgan：GFPGANv1.2 / v1.3（spandrel 0.4.2 支持，scale=1、min 512、不支持 fp16）
- restoreformer：RestoreFormer（spandrel 0.4.2 支持，size 需 32 的倍数）
**CodeFormer 不在 spandrel 0.4.2（当前最新发布版）里**，故本算子不提供（§48.7）。

检测：insightface antelopev2（INSIGHTFACE_HOME，默认 /models/insightface）——
与 instantid 族同款模型包，onnxruntime CPU 推理，不占显存。

口径（§48.6 A6）：**只改人脸区域**——裁剪框外的像素逐字节保留；
贴回用椭圆形羽化蒙版，避免硬边接缝。修复块按模型要求对齐（GFPGAN ≥512，
RestoreFormer 32 的倍数），`upscale` 先把块放大再修复、再缩回原块尺寸贴回，
以此换取更细的修复纹理。

分发：本文件是节点共享算子族（单一事实源），由 nodes/_tools/sync-common.py
分发为成员节点的 server_op.py，节点运行时经 ensure_plugin 自注册。
依赖：spandrel / insightface / torch / numpy / cv2(headless)——一律函数内懒加载。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
OP_VERSION = "30ce33c38ed6"
OP_FAMILY = "face_restore"  # 必须等于本文件 stem（gate J）


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

import os

from PIL import Image, ImageDraw, ImageFilter

FACERESTORE_DIR = os.environ.get("FACERESTORE_MODELS_DIR", "/models/facerestore")
INSIGHTFACE_HOME = os.environ.get("INSIGHTFACE_HOME", "/models/insightface")

# 模型名 → (权重文件名候选, size 倍数要求, 最小边长)
MODELS = {
    "gfpgan": (("GFPGANv1.3.pth", "GFPGANv1.2.pth", "GFPGAN.pth"), 32, 512),
    "restoreformer": (("RestoreFormer.pth", "RestoreFormer_plus.pth"), 32, 512),
}

_MODELS_CACHE = {}   # (model, path) -> spandrel ImageModelDescriptor
_FACE_APP = None


def _model_path(model, op_name):
    """定位修复模型权重；缺文件时报错点名该放哪。"""
    if model not in MODELS:
        raise ValueError(f"{op_name}: model 非法 {model!r}，可选 {tuple(MODELS)}"
                         f"（CodeFormer 不在 spandrel 0.4.2，见 dev-plan §48.7）")
    for fn in MODELS[model][0]:
        p = os.path.join(FACERESTORE_DIR, fn)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"{op_name}: 人脸修复权重缺失：{FACERESTORE_DIR}/{MODELS[model][0][0]}"
        f"（可选文件名 {'/'.join(MODELS[model][0])}；下载见 nodes/README.md 权重清单）")


def _restorer(model, op_name):
    """spandrel 加载修复模型（进程级单例，fp32——GFPGAN/RestoreFormer 都不支持 fp16）。"""
    path = _model_path(model, op_name)   # 先快速失败（FileNotFoundError）
    key = (model, path)
    if key in _MODELS_CACHE:
        return _MODELS_CACHE[key]
    import torch  # 懒加载
    from spandrel import ModelLoader
    if not torch.cuda.is_available():
        raise RuntimeError(f"{op_name}: 需要 CUDA（人脸修复为 GPU 推理），"
                           f"当前 torch.cuda.is_available()=False")
    desc = ModelLoader(device="cpu").load_from_file(path)
    desc.to("cuda").eval()
    _MODELS_CACHE[key] = desc
    print(f"[{op_name}] 修复模型已加载：{os.path.basename(path)} "
          f"arch={type(desc).__name__} scale={getattr(desc, 'scale', '?')}",
          flush=True)
    return desc


def _face_app(op_name):
    """insightface 检测器（进程级单例，CPU 推理）。"""
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_HOME,
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _FACE_APP = app
        print(f"[{op_name}] insightface antelopev2 ready "
              f"(home={INSIGHTFACE_HOME})", flush=True)
    return _FACE_APP


def _detect(img, op_name, det_thresh):
    """→ [(bbox, area)]，按面积降序。"""
    import cv2
    import numpy as np
    app = _face_app(op_name)
    arr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = app.get(arr)
    out = []
    for f in faces:
        x0, y0, x1, y1 = [float(v) for v in f.bbox]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area <= 0 or float(getattr(f, "det_score", 1.0)) < float(det_thresh):
            continue
        out.append(((x0, y0, x1, y1), area))
    out.sort(key=lambda t: t[1], reverse=True)
    return out


def _pick(faces, w, h, only_center, max_faces):
    """选要处理的脸：only_center → 离画面中心最近；否则面积最大的若干张。"""
    if only_center and faces:
        cx, cy = w / 2.0, h / 2.0
        def dist(t):
            x0, y0, x1, y1 = t[0]
            return ((x0 + x1) / 2 - cx) ** 2 + ((y0 + y1) / 2 - cy) ** 2
        faces = sorted(faces, key=dist)
    return faces[:max(1, int(max_faces))]


def _crop_box(bbox, pad, w, h):
    x0, y0, x1, y1 = bbox
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * pad, bh * pad
    return (max(0, int(x0 - px)), max(0, int(y0 - py)),
            min(w, int(x1 + px)), min(h, int(y1 + py)))


def _ellipse_mask(size, feather):
    """羽化椭圆蒙版：中心 255，边缘 feather 像素过渡。"""
    w, h = size
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    inset = max(1, int(min(w, h) * 0.06))
    d.ellipse((inset, inset, w - inset, h - inset), fill=255)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(float(feather)))
    return m


def _restore_crop(crop, model, upscale, op_name):
    """单块修复：按模型要求对齐边长 → 前向 → 缩回块原尺寸。"""
    import numpy as np
    import torch
    _, multiple, minimum = MODELS[model]
    desc = _restorer(model, op_name)
    cw, ch = crop.size
    # GFPGAN 的 style_mlp 输入维度写死在 512（喂 640 会报
    # `mat1 and mat2 shapes cannot be multiplied (1x6400 and 4096x8192)`），
    # 故 GFPGAN 固定 512×512；RestoreFormer 是全卷积，可按 32 倍数放大。
    side = int(round(minimum * max(1.0, float(upscale))))
    side = ((side + multiple - 1) // multiple) * multiple
    if model == "gfpgan" and side != minimum:
        raise ValueError(
            f"{op_name}: GFPGAN 只支持 {minimum}×{minimum} 输入（style_mlp 形状固定），"
            f"upscale 必须为 1.0；要更大尺度请用 model=\"restoreformer\"")
    # 一律方形成像（官方 FaceRestoreHelper 也是 align+warp 到 512²）
    side = int(side)
    x = crop.convert("RGB").resize((side, side), Image.LANCZOS)
    arr = np.asarray(x, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        y = desc(t)
    y = y.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    out = Image.fromarray((y * 255.0).round().astype(np.uint8), "RGB")
    return out.resize((cw, ch), Image.LANCZOS)


def face_restore(image, model="gfpgan", upscale=1.0, padding=0.3, feather=16,
                 only_center=True, max_faces=1, det_thresh=0.5):
    """人脸修复：IMAGE → IMAGE（只改人脸区域，裁剪框外逐字节保留）。

    - model：gfpgan / restoreformer
    - upscale：模型输入边长倍数（GFPGAN **必须 1.0**（512² 固定）；RestoreFormer 可 2.0/4.0）
    - padding：裁剪框外扩比例（0.3 = 各向外扩 30%）
    - feather：贴回蒙版羽化像素（0 = 硬边）
    - only_center：只修离画面中心最近的一张脸；max_faces：最多修几张
    """
    op_name = f"face.restore@{OP_VERSION}"
    if not isinstance(image, Image.Image):
        raise ValueError(f"{op_name}: image 需为 IMAGE 对象，got "
                         f"{type(image).__name__}")
    upscale = float(upscale)
    if upscale < 1.0 or upscale > 4.0:
        raise ValueError(f"{op_name}: upscale 需在 1.0~4.0，got {upscale}")
    if float(padding) < 0 or float(padding) > 2.0:
        raise ValueError(f"{op_name}: padding 需在 0~2，got {padding}")
    # 参数级校验放在加载模型之前（fail fast；也让本地无 GPU 能测这段契约）
    if model not in MODELS:
        raise ValueError(f"{op_name}: model 非法 {model!r}，可选 {tuple(MODELS)}"
                         f"（CodeFormer 不在 spandrel 0.4.2，见 dev-plan §48.7）")
    if model == "gfpgan" and float(upscale) != 1.0:
        raise ValueError(
            f"{op_name}: GFPGAN 只支持 512×512 输入（style_mlp 形状固定，喂 640 会报"
            f" shapes cannot be multiplied），upscale 必须为 1.0；"
            f"要更大尺度请用 model='restoreformer'")

    # 先查权重与检测器（缺权重/无人脸时报可操作的错），再进推理
    _restorer(model, op_name)      # 权重定位（FileNotFoundError 先于 CUDA 报错）
    src = image.convert("RGB")
    faces = _detect(src, op_name, det_thresh)
    if not faces:
        raise ValueError(f"{op_name}: 未检测到人脸（换张清晰正脸，"
                         f"或调低 det_thresh，当前 {det_thresh}）")
    picked = _pick(faces, src.width, src.height, only_center, max_faces)

    out = src.copy()
    boxes = []
    for bbox, _ in picked:
        box = _crop_box(bbox, float(padding), src.width, src.height)
        if box[2] - box[0] < 16 or box[3] - box[1] < 16:
            continue
        crop = src.crop(box)
        fixed = _restore_crop(crop, model, upscale, op_name)
        out.paste(fixed, (box[0], box[1]),
                  _ellipse_mask(crop.size, float(feather)))
        boxes.append(box)

    print(f"[{op_name}] {src.size} model={model} upscale={upscale} "
          f"检出 {len(faces)} 张 / 修复 {len(boxes)} 张 框={boxes}", flush=True)
    return {"image": out, "faces": len(boxes)}


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "face.restore",
        inputs={"image": "IMAGE", "model": "STRING", "upscale": "FLOAT",
                "padding": "FLOAT", "feather": "INT", "only_center": "BOOL",
                "max_faces": "INT", "det_thresh": "FLOAT"},
        outputs={"image": "IMAGE", "faces": "INT"},
        description="Face Restore（G6）：检测人脸 → 裁剪 → spandrel 修复（GFPGAN/"
                    "RestoreFormer）→ 羽化贴回。只改人脸区域，框外像素逐字节保留；"
                    "权重在 FACERESTORE_MODELS_DIR（默认 /models/facerestore）"
    )(face_restore)
