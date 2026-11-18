"""video.face_detail：视频级 SD 逐帧修脸（ADetailer 式），可选 ControlNet tile 锁结构。

背景（2026-11-18）：H3/Wan 生在「大幅运动 + 小脸」场景会面部时序崩坏。
GFPGAN（video.enhance）能修清晰度，但 face_weight 调高就变通用网红脸（身份漂移）。
本算子走 SD 局部 img2img 路线（对标 ComfyUI FaceDetailer）：

  视频 → 逐帧 YOLO 人脸检测（face_yolov8n）→ 检测框时序 EMA 平滑 + 缺帧近邻填补
       → 裁剪外扩 padding → 短边放大到 guide_size（对齐 8）
       → [可选] ControlNet tile 以裁剪自身为 hint（锁轮廓，防「修成另一个人」）
       → vae_encode → SD img2img（低 denoise 0.30~0.40）→ vae_decode
       → resize 回原裁剪尺寸 → 羽化贴回原帧 → 输出视频

时序一致性手段（逐帧图像模型的通病：帧间跳变）：
- 检测框 EMA 平滑（temporal_alpha，0=不平滑/1=完全冻结上一帧）
- 固定 seed（每帧同种子 ⇒ 同输入近似同输出，配合低 denoise 抖动小）
- 缺检帧用前帧框（不跳位置）；整段零检测时原样透传

模型：
- checkpoint：MODELS_DIR 下 checkpoint（如 Realistic_Vision_V5.1 / sd_xl_base_1.0）
- detector：DETECTOR_DIR 下 YOLO（face_yolov8n/s、hand_yolov8n）
- controlnet：MODELS_DIR 下 controlnet（如 control_v11f1e_sd15_tile_fp16），空串关闭
"""
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
OP_VERSION = "7ee3f4e27480"
OP_FAMILY = "sd_face"  # 必须等于本文件 stem（gate J）


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)


import os
import time

from PIL import Image, ImageDraw, ImageFilter

from app import ops

DETECTOR_DIR = os.environ.get("DETECTOR_DIR", "/models/detectors")
_DETECTOR_CACHE = {}
_CN_CACHE = {}


# ---------------------------------------------------------------------------
# 基础设施：检测器 / 视频解码
# ---------------------------------------------------------------------------

def _detector(name):
    """YOLO 检测器缓存（ultralytics 惰性导入）。"""
    if name not in _DETECTOR_CACHE:
        path = os.path.join(DETECTOR_DIR, name + ".pt")
        if not os.path.isfile(path):
            import glob
            hits = sorted(glob.glob(os.path.join(DETECTOR_DIR, name + "*.pt")))
            if not hits:
                available = (sorted(os.listdir(DETECTOR_DIR))
                             if os.path.isdir(DETECTOR_DIR) else [])
                raise FileNotFoundError(
                    f"detector '{name}' not found: {path}; "
                    f"available: {available or '(empty)'}")
            path = hits[0]
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "video.face_detail 需要 ultralytics（pip install ultralytics）"
            ) from e
        _DETECTOR_CACHE[name] = YOLO(path)
    return _DETECTOR_CACHE[name]


def _decode_video(video, op_name):
    """VIDEO 对象 → (PIL 帧列表, fps)；cv2 解码绕行系统 ffmpeg（存储劣化时会卡）。"""
    if isinstance(video, dict) and video.get("frames"):
        return list(video["frames"]), float(video.get("fps") or 16)
    path = video.get("path") if isinstance(video, dict) else None
    if not (path and os.path.isfile(path)):
        raise ValueError(f"{op_name}: video 需为 VIDEO 对象（frames 或 path），"
                         f"got {str(video)[:120]}")
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"{op_name}: cv2 打不开视频 {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 16)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise ValueError(f"{op_name}: 视频无有效帧 {path}")
    print(f"[{op_name}] cv2 解码 {len(frames)} 帧 {frames[0].width}x"
          f"{frames[0].height} @{fps:.2f}fps", flush=True)
    return frames, fps


def _detect_boxes(frames, detector, conf, max_targets, op_name):
    """逐帧检测 → (boxes, confs)；boxes 为 [(x1,y1,x2,y2)]，无检帧为空列表。"""
    yolo = _detector(detector)
    t0 = time.time()
    per_frame = []
    for i, f in enumerate(frames):
        res = yolo.predict(f, conf=conf, verbose=False)
        b = res[0].boxes
        if b is None or len(b) == 0:
            per_frame.append(([], []))
            continue
        xyxy = b.xyxy.tolist()
        confs = b.conf.tolist()
        order = sorted(range(len(xyxy)), key=lambda k: -confs[k])[:max_targets]
        per_frame.append(([xyxy[k] for k in order], [confs[k] for k in order]))
        if (i + 1) % 25 == 0 or i + 1 == len(frames):
            print(f"[{op_name}] 检测 {i + 1}/{len(frames)} 帧 "
                  f"({time.time() - t0:.1f}s)", flush=True)
    return per_frame


def _temporal_smooth(per_frame, alpha, op_name):
    """检测框时序平滑：缺帧继承上一帧 + EMA 混合（alpha 越大越平滑）。"""
    alpha = max(0.0, min(1.0, float(alpha)))
    out = []
    prev = None
    n_fill = n_miss = 0
    for boxes, confs in per_frame:
        if not boxes:
            n_miss += 1
            if prev is not None:
                out.append(([list(b) for b in prev], [0.0] * len(prev)))
                n_fill += 1
            else:
                out.append(([], []))
            continue
        cur = [list(b) for b in boxes]
        if prev is not None and len(prev) == len(cur) and alpha > 0:
            cur = [[alpha * p + (1 - alpha) * c for p, c in zip(pb, cb)]
                   for pb, cb in zip(prev, cur)]
        out.append((cur, list(confs)))
        prev = cur
    print(f"[{op_name}] 检测框平滑 alpha={alpha} 缺检 {n_miss} 帧，"
          f"近邻填补 {n_fill} 帧", flush=True)
    return out


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------

def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    def video_face_detail(video, ckpt="Realistic_Vision_V5.1", prompt="",
                          negative="", detector="face", det_conf=0.5,
                          padding=0.4, denoise=0.35, steps=20, cfg=6.0,
                          guide_size=512, feather=16, max_targets=1,
                          seed=20261116, sampler_name="euler",
                          scheduler="normal", controlnet="", cn_strength=0.6,
                          temporal_alpha=0.6, fps=0):
        """视频级 SD 修脸：逐帧检测→裁剪→低 denoise img2img→羽化贴回。

        - denoise：0.30~0.40 推荐。低=保身份（纹理改善有限），高=修得狠但会换脸
        - temporal_alpha：检测框 EMA 平滑（0 关闭；0.6 稳位且不拖影）
        - controlnet：MODELS_DIR 下 controlnet 名（control_v11f1e_sd15_tile_fp16）；
          空串关闭。开启后以人脸裁剪自身为 hint 锁轮廓（治「修成另一个人」）
        - prompt/negative：应描述**目标脸**特征；空 prompt 时纯粹按原 latent 修
        """
        op_name = f"video.face_detail@{OP_VERSION}"
        t_all = time.time()

        frames, in_fps = _decode_video(video, op_name)
        if not frames:
            raise ValueError(f"{op_name}: 帧列表为空")

        # 1) 检测 + 时序平滑
        per_frame = _detect_boxes(frames, detector, float(det_conf),
                                  int(max_targets), op_name)
        if not any(b for b, _ in per_frame):
            print(f"[{op_name}] 整段未检出 {detector}，原样透传", flush=True)
            return {"video": {"frames": frames,
                              "fps": int(fps) if int(fps) > 0 else in_fps,
                              "num_frames": len(frames)}, "faces": 0}
        per_frame = _temporal_smooth(per_frame, temporal_alpha, op_name)

        # 2) 模型装配（checkpoint 走 model_manager，幂等命中缓存）
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚未就绪
        loaded = ops.checkpoint_load(models, ckpt)
        pipe, _ = ops.resolve_pipe(loaded["model"])
        model = loaded["model"]
        pos = ops.clip_encode(pipe, prompt or "", 0, 0)["cond"]
        neg = ops.clip_encode(pipe, negative or "", 0, 0)["cond"]
        print(f"[{op_name}] ckpt={ckpt} prompt={str(prompt)[:60]!r} "
              f"denoise={denoise} steps={steps} guide={guide_size}", flush=True)

        # 3) ControlNet（可选）
        control_net = None
        cn_name = str(controlnet or "").strip()
        if cn_name:
            key = f"cn:{cn_name}"
            if key not in _CN_CACHE:
                _, _ = models.load_controlnet(cn_name)
                _CN_CACHE[key] = models.get(key)
            control_net = _CN_CACHE[key]
            print(f"[{op_name}] controlnet={cn_name} strength={cn_strength}",
                  flush=True)

        # 4) 逐帧重绘
        out, n_hit = [], 0
        for i, (img, (boxes, confs)) in enumerate(zip(frames, per_frame)):
            if not boxes:
                out.append(img)
                continue
            canvas = img.copy()
            W, H = canvas.size
            x1, y1, x2, y2 = boxes[0]
            bw, bh = x2 - x1, y2 - y1
            px, py = bw * float(padding), bh * float(padding)
            cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))
            if cx2 - cx1 < 32 or cy2 - cy1 < 32:
                out.append(canvas)
                continue
            crop = canvas.crop((cx1, cy1, cx2, cy2))
            scale = float(guide_size) / min(crop.size)
            tw = max(64, round(crop.width * scale) // 8 * 8)
            th = max(64, round(crop.height * scale) // 8 * 8)
            crop_big = crop.resize((tw, th), Image.Resampling.LANCZOS)

            control = None
            if control_net is not None:
                control = ops.controlnet_apply(control_net, crop_big,
                                               float(cn_strength))
            lat = ops.vae_encode(pipe, crop_big)["latent"]
            res = ops.sample(model, pos, neg, lat, seed=int(seed),
                             steps=int(steps), cfg=float(cfg),
                             sampler_name=sampler_name, scheduler=scheduler,
                             denoise=float(denoise), control=control)
            refined = ops.vae_decode(pipe, res["latent"])["image"]
            refined = refined.resize(crop.size, Image.Resampling.LANCZOS)

            mask = Image.new("L", crop.size, 0)
            f_pad = max(1, int(feather))
            ImageDraw.Draw(mask).rectangle(
                [f_pad, f_pad, crop.width - f_pad, crop.height - f_pad],
                fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(f_pad / 2))
            canvas.paste(refined, (cx1, cy1), mask)
            out.append(canvas)
            n_hit += 1
            if (i + 1) % 10 == 0 or i + 1 == len(frames):
                print(f"[{op_name}] 重绘 {i + 1}/{len(frames)} 帧 "
                      f"({time.time() - t_all:.1f}s)", flush=True)

        print(f"[{op_name}] done {len(out)}帧 修脸 {n_hit} 帧 "
              f"{time.time() - t_all:.1f}s", flush=True)
        return {"video": {"frames": out,
                          "fps": int(fps) if int(fps) > 0 else in_fps,
                          "num_frames": len(out)}, "faces": n_hit}

    registry.register(
        "video.face_detail",
        inputs={"video": "VIDEO", "ckpt": "STRING", "prompt": "STRING",
                "negative": "STRING", "detector": "STRING",
                "det_conf": "FLOAT", "padding": "FLOAT", "denoise": "FLOAT",
                "steps": "INT", "cfg": "FLOAT", "guide_size": "INT",
                "feather": "INT", "max_targets": "INT", "seed": "INT",
                "sampler_name": "STRING", "scheduler": "STRING",
                "controlnet": "STRING", "cn_strength": "FLOAT",
                "temporal_alpha": "FLOAT", "fps": "INT"},
        outputs={"video": "VIDEO", "faces": "INT"},
        description="视频级 SD 修脸（ADetailer/FaceDetailer 式）：逐帧 YOLO 检测→"
                    "裁剪放大→低 denoise img2img→羽化贴回。检测框 EMA 时序平滑 + "
                    "固定 seed 抑制帧间跳变。controlnet（tile）可选，用裁剪自身作 hint "
                    "锁轮廓，治「修成另一个人」")
