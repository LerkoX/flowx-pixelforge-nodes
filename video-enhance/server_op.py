"""video.enhance：视频二次修复（逐帧面部修复 + AI 超分）。

Animate/生成类视频的后处理一条龙：视频 → 逐帧（insightface 检测人脸 →
GFPGAN/RestoreFormer 修复 → 椭圆羽化贴回）→ 逐帧（RealESRGAN/SwinIR
模型超分）→ 视频。两阶段独立开关（face_model="none" / upscale_model=""）。

⚠ 逐帧图像模型固有限制（2026-11-15 用户沟通）：无帧间时序建模，
高速运动+小脸场景可能轻微闪烁；要更强时序一致性需 BasicVSR 级模型
（未接入）。实践中确定性推理让相似帧输出相似，静态区域基本不闪。

模型目录（环境变量可覆盖，缺文件时报错点名放哪）：
- 面部修复：FACERESTORE_MODELS_DIR（默认 /models/facerestore）
  gfpgan=GFPGANv1.3/v1.2（512² 固定）；restoreformer=RestoreFormer（32 倍数）
- 超分：UPSCALE_MODELS_DIR（默认 /models/upscale）
  RealESRGAN_x2plus/x4plus、SwinIR 等 .pth/.safetensors（spandrel 自动识别）
- 检测器：INSIGHTFACE_HOME（默认 /models/insightface，antelopev2）

实现参考 face_restore.py / upscale.py（同仓族文件），逻辑内联复制——
服务端插件按族文件独立加载，跨族 import 不可靠，故自包含。
"""
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
OP_VERSION = "bda9125814b0"
OP_FAMILY = "video_enhance"  # 必须等于本文件 stem（gate J）


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
import time

from PIL import Image, ImageDraw, ImageFilter

FACERESTORE_DIR = os.environ.get("FACERESTORE_MODELS_DIR", "/models/facerestore")
UPSCALE_DIR = os.environ.get("UPSCALE_MODELS_DIR", "/models/upscale")
INSIGHTFACE_HOME = os.environ.get("INSIGHTFACE_HOME", "/models/insightface")

# 模型名 → (权重文件名候选, size 倍数要求, 最小边长)
FACE_MODELS = {
    "gfpgan": (("GFPGANv1.3.pth", "GFPGANv1.2.pth", "GFPGAN.pth"), 32, 512),
    "restoreformer": (("RestoreFormer.pth", "RestoreFormer_plus.pth"), 32, 512),
}
_UP_EXTS = (".pth", ".safetensors", ".pt", ".ckpt")

_FACE_DESC_CACHE = {}
_UP_CACHE = {}
_FACE_APP = None


def _face_model_path(model, op_name):
    if model not in FACE_MODELS:
        raise ValueError(f"{op_name}: face_model 非法 {model!r}，"
                         f"可选 {tuple(FACE_MODELS)} 或 none")
    for fn in FACE_MODELS[model][0]:
        p = os.path.join(FACERESTORE_DIR, fn)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"{op_name}: 人脸修复权重缺失：{FACERESTORE_DIR}/"
        f"{FACE_MODELS[model][0][0]}")


def _face_restorer(model, op_name):
    path = _face_model_path(model, op_name)
    if path in _FACE_DESC_CACHE:
        return _FACE_DESC_CACHE[path]
    import torch
    from spandrel import ModelLoader
    if not torch.cuda.is_available():
        raise RuntimeError(f"{op_name}: 需要 CUDA")
    desc = ModelLoader(device="cpu").load_from_file(path)
    desc.to("cuda").eval()   # GFPGAN/RestoreFormer 不支持 fp16，保持 fp32
    _FACE_DESC_CACHE[path] = desc
    print(f"[{op_name}] 面部修复模型已加载：{os.path.basename(path)}", flush=True)
    return desc


def _face_app(op_name):
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="antelopev2", root=INSIGHTFACE_HOME,
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _FACE_APP = app
        print(f"[{op_name}] insightface antelopev2 ready", flush=True)
    return _FACE_APP


def _ellipse_mask(size, feather):
    w, h = size
    m = Image.new("L", size, 0)
    d = ImageDraw.Draw(m)
    inset = max(1, int(min(w, h) * 0.06))
    d.ellipse((inset, inset, w - inset, h - inset), fill=255)
    if feather > 0:
        m = m.filter(ImageFilter.GaussianBlur(float(feather)))
    return m


def _face_restore_one(img, face_model, det_thresh, padding, feather, weight,
                      op_name):
    """单帧面部修复：检出最大一张脸 → 512² 修复 → 羽化贴回。无脸则原样返回。"""
    import cv2
    import numpy as np
    import torch
    app = _face_app(op_name)
    desc = _face_restorer(face_model, op_name)
    src = img.convert("RGB")
    arr = cv2.cvtColor(np.asarray(src), cv2.COLOR_RGB2BGR)
    faces = app.get(arr)
    best, best_area = None, 0.0
    for f in faces:
        x0, y0, x1, y1 = [float(v) for v in f.bbox]
        area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
        if area > best_area and float(getattr(f, "det_score", 1.0)) >= det_thresh:
            best, best_area = (x0, y0, x1, y1), area
    if best is None:
        return src, 0
    x0, y0, x1, y1 = best
    bw, bh = x1 - x0, y1 - y0
    px, py = bw * padding, bh * padding
    box = (max(0, int(x0 - px)), max(0, int(y0 - py)),
           min(src.width, int(x1 + px)), min(src.height, int(y1 + py)))
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return src, 0
    crop = src.crop(box)
    _, _, minimum = FACE_MODELS[face_model]
    cw, ch = crop.size
    x = crop.resize((minimum, minimum), Image.LANCZOS)
    a = np.asarray(x, dtype=np.float32) / 255.0
    t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        y = desc(t)
    y = y.clamp(0, 1).squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    fixed = Image.fromarray((y * 255.0).round().astype("uint8"), "RGB")
    fixed = fixed.resize((cw, ch), Image.LANCZOS)
    # **身份保持混合（2026-11-15 成片 243 复盘）**：GFPGAN 全量贴回会把脸
    # 修成自己的训练先验（身份漂移：不像参考图）+ 与超分锐化叠加过锐。
    # weight<1 时按 fixed×w + 原crop×(1-w) 混合，保身份减修复感。
    w = max(0.0, min(1.0, float(weight)))
    if w < 1.0:
        import numpy as _np
        fb = _np.asarray(fixed, dtype=_np.float32)
        ob = _np.asarray(crop.convert("RGB"), dtype=_np.float32)
        fixed = Image.fromarray(
            (fb * w + ob * (1.0 - w)).round().clip(0, 255).astype("uint8"),
            "RGB")
    out = src.copy()
    out.paste(fixed, (box[0], box[1]), _ellipse_mask(crop.size, feather))
    return out, 1


def _upscale_path(name, op_name):
    if os.path.basename(name) != name:
        raise ValueError(f"{op_name}: upscale_model 需为纯文件名，got {name!r}")
    for ext in _UP_EXTS:
        p = os.path.join(UPSCALE_DIR, name + ext)
        if os.path.isfile(p):
            return p
    # 允许直接给带扩展名的完整文件名
    p = os.path.join(UPSCALE_DIR, name)
    if os.path.isfile(p):
        return p
    raise FileNotFoundError(
        f"{op_name}: 超分模型缺失：{UPSCALE_DIR}/{name}（{_UP_EXTS}）")


def _upscale_handle(name, op_name):
    path = _upscale_path(name, op_name)
    if path in _UP_CACHE:
        return _UP_CACHE[path]
    import torch
    from spandrel import ModelLoader
    model = ModelLoader(device="cpu").load_from_file(path)
    model.to("cuda")
    dtype = torch.float32
    try:
        model.to(torch.float16)
        dtype = torch.float16
    except Exception:  # noqa: BLE001
        pass
    model.eval()
    scale = int(getattr(model, "scale", 2))
    _UP_CACHE[path] = (model, dtype, scale)
    print(f"[{op_name}] 超分模型已加载：{os.path.basename(path)} "
          f"scale={scale} dtype={dtype}", flush=True)
    return _UP_CACHE[path]


def _upscale_one(img, handle, tile, op_name):
    """单帧超分：tile=0 整图；tile>0 分块（overlap 16px）拼接。"""
    import numpy as np
    import torch
    model, dtype, scale = handle
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda().to(dtype)

    def _fwd(t):
        with torch.no_grad():
            y = model(t)
        return y.clamp(0, 1).float()

    tile = int(tile)
    if tile <= 0:
        y = _fwd(x)
    else:
        overlap = 16
        _, _, H, W = x.shape
        y = torch.zeros((1, 3, H * scale, W * scale), dtype=torch.float32)
        wsum = torch.zeros((1, 1, H * scale, W * scale), dtype=torch.float32)
        step = tile - overlap
        for ty in range(0, H, step):
            for tx in range(0, W, step):
                patch = x[:, :, ty:ty + tile, tx:tx + tile]
                ph, pw = patch.shape[2], patch.shape[3]
                out = _fwd(patch).cpu()
                y[:, :, ty * scale:(ty + ph) * scale,
                  tx * scale:(tx + pw) * scale] += out
                wsum[:, :, ty * scale:(ty + ph) * scale,
                     tx * scale:(tx + pw) * scale] += 1.0
        y = (y / wsum.clamp(min=1e-8)).cuda()
    y = y.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((y * 255.0).round().astype("uint8"), "RGB")


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    def video_enhance(video, face_model="gfpgan", upscale_model="",
                      det_thresh=0.5, face_padding=0.3, face_feather=16,
                      face_weight=0.5, tile=0, fps=0):
        """视频二次修复：逐帧（面部修复 → 模型超分）。

        - face_model：gfpgan / restoreformer / none（跳过面部修复）
        - upscale_model：UPSCALE_MODELS_DIR 下模型文件名（可不带扩展名）；
          空串跳过超分。倍率由模型自身 scale 决定（x2plus=2×）
        - tile：超分分块边长（0=整图；大分辨率显存不足时设 512/256）
        - fps：输出帧率（0=跟随输入）
        """
        op_name = f"video.enhance@{OP_VERSION}"
        import time as _t

        # 输入统一成 PIL 帧列表——兼容两种对象形态（同 video_mask.py 约定）：
        # {"frames": [...], "fps": ...}（内存态）与 {"path": ...}（落盘态，
        # 跨 job 传递后 frames 会被物化成文件，走 ffmpeg 裸流解码）。
        if isinstance(video, dict) and video.get("frames"):
            frames = list(video["frames"])
            in_fps = float(video.get("fps") or 16)
        elif isinstance(video, dict) and video.get("path") \
                and os.path.isfile(video["path"]):
            # **OpenCV 解码（2026-11-15 环境事故绕行）**：系统 ffmpeg/ffprobe
            # 二进制在 compshare 存储后端劣化时启动即卡死（连 -version 都
            # 阻塞）；cv2 是 pip 静态链接自带 ffmpeg 库，不走系统二进制。
            import cv2
            path = video["path"]
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"{op_name}: cv2 打不开视频 {path}")
            in_fps = float(cap.get(cv2.CAP_PROP_FPS) or 16)
            frames = []
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                frames.append(Image.fromarray(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            cap.release()
            if not frames:
                raise ValueError(f"{op_name}: 视频无有效帧 {path}")
            vw, vh = frames[0].size
            print(f"[{op_name}] path 形态 cv2 解码 {len(frames)} 帧 "
                  f"{vw}x{vh} @{in_fps:.2f}fps", flush=True)
        else:
            raise ValueError(f"{op_name}: video 需为 VIDEO 对象"
                             f"（frames 或 path），got "
                             f"{str(video)[:120]}")
        face_model = str(face_model or "none").lower()
        upscale_model = str(upscale_model or "").strip()
        if face_model not in ("none",) and face_model not in FACE_MODELS:
            raise ValueError(f"{op_name}: face_model 非法 {face_model!r}，"
                             f"可选 {tuple(FACE_MODELS)} 或 none")

        up_handle = _upscale_handle(upscale_model, op_name) \
            if upscale_model else None
        if face_model != "none":
            _face_restorer(face_model, op_name)   # fail fast（缺权重先报错）

        out, n_face = [], 0
        t0 = _t.time()
        for i, f in enumerate(frames):
            img = f if isinstance(f, Image.Image) else Image.fromarray(f)
            if face_model != "none":
                img, hit = _face_restore_one(img, face_model, det_thresh,
                                             float(face_padding),
                                             int(face_feather),
                                             float(face_weight), op_name)
                n_face += hit
            if up_handle is not None:
                img = _upscale_one(img, up_handle, int(tile), op_name)
            out.append(img)
            if (i + 1) % 10 == 0 or i + 1 == len(frames):
                print(f"[{op_name}] {i + 1}/{len(frames)} 帧 "
                      f"({_t.time() - t0:.1f}s)", flush=True)

        scale = up_handle[2] if up_handle else 1
        print(f"[{op_name}] done {len(out)}帧 面部修复 {n_face} 帧命中 "
              f"scale={scale} {_t.time() - t0:.1f}s", flush=True)
        return {"video": {"frames": out,
                          "fps": int(fps) if int(fps) > 0 else in_fps,
                          "num_frames": len(out)},
                "faces": n_face, "scale": scale}

    registry.register(
        "video.enhance",
        inputs={"video": "VIDEO", "face_model": "STRING",
                "upscale_model": "STRING", "det_thresh": "FLOAT",
                "face_padding": "FLOAT", "face_feather": "INT",
                "face_weight": "FLOAT", "tile": "INT", "fps": "INT"},
        outputs={"video": "VIDEO", "faces": "INT", "scale": "INT"},
        description="视频二次修复：逐帧 GFPGAN/RestoreFormer 面部修复"
                    "（羽化贴回只改人脸）+ RealESRGAN/SwinIR 模型超分；"
                    "两阶段独立开关。权重在 /models/facerestore 与 "
                    "/models/upscale。注意逐帧处理无帧间时序建模，"
                    "高速运动场景可能轻微闪烁。"
    )(video_enhance)
