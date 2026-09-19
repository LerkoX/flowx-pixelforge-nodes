"""detail-refine 节点的服务端算子插件（随节点包自注册，经 POST /admin/plugins 上传）。

ADetailer 式局部重绘：YOLO 检测脸/手 → 裁剪外扩 → 放大到 guide_size →
img2img 重绘（denoise<1）→ 羽化贴回原图。修脸/修手专用。
依赖服务端核心原语：app.ops 的 resolve_pipe / vae_encode / vae_decode / sample。
依赖容器内 pip 包：ultralytics（见 requirements-pascal.txt 尾部注释）。
"""
import glob
import os

from PIL import Image, ImageDraw, ImageFilter

from app import ops

DETECTOR_DIR = os.environ.get("DETECTOR_DIR", "/models/detectors")
_DETECTOR_CACHE = {}


def _detector(name):
    """YOLO 检测器缓存（ultralytics 惰性导入，未装时报错提示）。"""
    if name not in _DETECTOR_CACHE:
        # 精确 <name>.pt，否则前缀匹配 <name>*.pt（如 face → face_yolov8n.pt）
        path = os.path.join(DETECTOR_DIR, name + ".pt")
        if not os.path.isfile(path):
            hits = sorted(glob.glob(os.path.join(DETECTOR_DIR, name + "*.pt")))
            if hits:
                path = hits[0]
        if not os.path.isfile(path):
            available = (sorted(os.listdir(DETECTOR_DIR))
                         if os.path.isdir(DETECTOR_DIR) else [])
            raise FileNotFoundError(
                f"detector '{name}' not found: {path}; "
                f"available: {available or '(empty)'}")
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise RuntimeError(
                "detail.refine 需要 ultralytics（容器内 pip install ultralytics）") from e
        _DETECTOR_CACHE[name] = YOLO(path)
    return _DETECTOR_CACHE[name]


def register(registry):
    @registry.register(
        "detail.refine",
        inputs={"model": "MODEL", "pos": "COND", "neg": "COND", "image": "IMAGE",
                "detector": "STRING", "conf": "FLOAT", "padding": "FLOAT",
                "denoise": "FLOAT", "steps": "INT", "cfg": "FLOAT",
                "sampler_name": "STRING", "seed": "INT", "guide_size": "INT",
                "max_targets": "INT", "feather": "INT"},
        outputs={"image": "IMAGE", "count": "INT"},
        description="ADetailer 式局部重绘：YOLO 检测（detector=face/hand，模型在 "
                    "DETECTOR_DIR）→ 裁剪外扩 padding → 放大到 guide_size → "
                    "img2img 重绘(denoise) → 羽化贴回原图。修脸/修手专用；"
                    "conditioning 复用主管线，负面 embedding 同样生效")
    def _op_detail_refine(model, pos, neg, image, detector="face", conf=0.3,
                          padding=0.4, denoise=0.4, steps=20, cfg=7.0,
                          sampler_name=ops.DEFAULT_SAMPLER, seed=-1,
                          guide_size=512, max_targets=4, feather=16):
        yolo = _detector(detector)
        results = yolo.predict(image, conf=conf, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print(f"[detail.refine] {detector}: no target detected", flush=True)
            return {"image": image, "count": 0}

        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        order = sorted(range(len(xyxy)), key=lambda i: -confs[i])[:max_targets]

        pipe, _ = ops.resolve_pipe(model)
        img = image.copy()
        W, H = img.size
        n = 0
        for i in order:
            x1, y1, x2, y2 = xyxy[i]
            bw, bh = x2 - x1, y2 - y1
            px, py = bw * padding, bh * padding
            cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))
            if cx2 - cx1 < 32 or cy2 - cy1 < 32:
                continue
            crop = img.crop((cx1, cy1, cx2, cy2))
            # 短边放大到 guide_size，对齐 8（VAE 要求）
            scale = guide_size / min(crop.size)
            tw = max(64, round(crop.width * scale) // 8 * 8)
            th = max(64, round(crop.height * scale) // 8 * 8)
            crop_big = crop.resize((tw, th), Image.Resampling.LANCZOS)
            # img2img 重绘
            lat = ops.vae_encode(pipe, crop_big)["latent"]
            out = ops.sample(model, pos, neg, lat, seed=seed, steps=steps,
                             cfg=cfg, sampler_name=sampler_name, denoise=denoise)
            refined = ops.vae_decode(pipe, out["latent"])["image"]
            refined = refined.resize(crop.size, Image.Resampling.LANCZOS)
            # 羽化 mask 贴回（边缘渐变避免拼接痕）
            mask = Image.new("L", crop.size, 0)
            ImageDraw.Draw(mask).rectangle(
                [feather, feather, crop.width - feather, crop.height - feather],
                fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(feather / 2))
            img.paste(refined, (cx1, cy1), mask)
            n += 1
            print(f"[detail.refine] {detector}#{n} conf={confs[i]:.2f} "
                  f"box=({cx1},{cy1},{cx2},{cy2}) -> {tw}x{th} redraw",
                  flush=True)
        return {"image": img, "count": n}
