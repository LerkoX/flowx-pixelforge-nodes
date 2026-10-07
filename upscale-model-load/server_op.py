"""高清放大模型算子（M6 UPSCALE_MODEL 链，插件化交付）。

- upscale_model.load：从 UPSCALE_MODELS_DIR（默认 /models）加载放大模型
  （spandrel 自动识别架构，支持 RealESRGAN/SwinIR 等 .pth/.safetensors），
  fp16 + cuda，进程级常驻 dict（模型小，~64MB 级，常驻代价可忽略）
- image.upscale_with_model：IMAGE → 模型放大 → IMAGE；tile>0 时按 tile×tile
  分块（overlap 16px）前向拼接，防 8GB 显存整图溢出（512→2048 整图 RRDBNet
  fp16 约 1.5~2GB 可过；更大输入请开 tile）

与 hires fix 的分工：本算子是**像素空间模型放大**（真实细节重建，4x 直出）；
hires fix 走 image.upscale/latent.upscale + sd.sample(denoise<1) 重采样精修
（构图锁定下的细节再生）。可串联：先模型放大保底，再低 denoise 精修。

分发：本文件是节点共享算子族（单一事实源），由 nodes/_tools/sync-common.py
分发为各成员节点的 server_op.py，节点运行时经 ensure_plugin 自注册到推理服务
（POST /admin/plugins，hash 幂等）。使用：对应专属节点或 inference-op 通用节点。

依赖：spandrel（烘焙进镜像）；torch/numpy/spandrel 一律函数内懒加载
（本地无 GPU 契约测试可导入）。
"""
import os

from PIL import Image

UPSCALE_DIR = os.environ.get("UPSCALE_MODELS_DIR", "/models")

_MODELS = {}  # resolved_path -> UpscaleHandle 进程级常驻

_EXTS = ("", ".pth", ".pt", ".safetensors", ".ckpt")


class UpscaleHandle:
    """UPSCALE_MODEL 注册表对象：模型 + 精度 + 倍率（进程内传递，不序列化）。
    spandrel ModelDescriptor 不透传 parameters()，输入侧无法可靠探测精度，
    故加载时确定 dtype 并随句柄携带（真机 t2 教训：fp16 模型 + fp32 输入
    会报 Input/bias type 不匹配）。"""

    def __init__(self, model, dtype, scale, path):
        self.model = model
        self.dtype = dtype      # torch.dtype
        self.scale = scale      # int 放大倍率（由模型架构决定）
        self.path = path


def _resolve(name):
    """模型名 → 文件路径（name 可带或不带扩展名）。"""
    if os.path.basename(name) != name:
        raise ValueError(f"upscale_model.load: name 需为纯文件名，got {name!r}")
    for ext in _EXTS:
        p = os.path.join(UPSCALE_DIR, name + ext)
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"upscale model '{name}' not found in {UPSCALE_DIR}（支持扩展名 {_EXTS}）")


def _get_model(name):
    path = _resolve(name)  # 纯 os 操作，先快速失败（FileNotFoundError）
    if path not in _MODELS:
        import torch  # 懒加载
        from spandrel import ModelLoader
        model = ModelLoader(device="cpu").load_from_file(path)
        model.to("cuda")
        dtype = torch.float32
        try:
            model.to(torch.float16)
            dtype = torch.float16
        except Exception as e:  # 个别架构 fp16 不兼容则退回 fp32
            print(f"[upscale_model.load] fp16 转换失败，退回 fp32: {e}",
                  flush=True)
        model.eval()
        scale = int(getattr(model, "scale", 4))
        _MODELS[path] = UpscaleHandle(model, dtype, scale, path)
        print(f"[upscale_model.load] {os.path.basename(path)} scale={scale} "
              f"dtype={dtype} resident={len(_MODELS)}", flush=True)
    return _MODELS[path]


def upscale_model_load(name):
    """加载像素放大模型：name 为 UPSCALE_MODELS_DIR 下的文件名（可不带扩展名）。
    返回 UPSCALE_MODEL 句柄供 image.upscale_with_model 使用。"""
    handle = _get_model(str(name))
    return {"upscale_model": handle}


def image_upscale_with_model(image, upscale_model, tile=0):
    """模型放大：IMAGE → IMAGE（倍率由模型自身 scale 决定，如 RealESRGAN_x4plus
    为 4x）。tile=0 整图前向；tile>0 分块（块边长，overlap 16px）拼接防显存溢出。
    """
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"image.upscale_with_model: image 需为 IMAGE 对象，"
            f"got {type(image).__name__}")
    if not isinstance(upscale_model, UpscaleHandle):
        raise ValueError(
            f"image.upscale_with_model: upscale_model 需为 upscale_model.load "
            f"的返回句柄，got {type(upscale_model).__name__}")
    import numpy as np  # 懒加载
    import torch

    model = upscale_model.model
    scale = upscale_model.scale

    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).cuda()
    x = x.to(upscale_model.dtype)  # 与模型精度一致（句柄携带）

    tile = int(tile)
    overlap = 16

    def _forward(chunk):
        with torch.no_grad():
            return model(chunk)

    if tile <= 0:
        y = _forward(x)
    else:
        _, _, h, w = x.shape
        y = torch.zeros((1, 3, h * scale, w * scale), dtype=x.dtype,
                        device=x.device)
        for ty in range(0, h, tile):
            for tx in range(0, w, tile):
                in_y0 = max(0, ty - overlap)
                in_x0 = max(0, tx - overlap)
                in_y1 = min(h, ty + tile + overlap)
                in_x1 = min(w, tx + tile + overlap)
                out = _forward(x[:, :, in_y0:in_y1, in_x0:in_x1])
                # 裁掉与已写区域重叠的部分（输入坐标的相对偏移 × scale）
                oy0 = (ty - in_y0) * scale
                ox0 = (tx - in_x0) * scale
                oy1 = out.shape[2] - (in_y1 - min(h, ty + tile)) * scale
                ox1 = out.shape[3] - (in_x1 - min(w, tx + tile)) * scale
                y[:, :, ty * scale:min(h, ty + tile) * scale,
                  tx * scale:min(w, tx + tile) * scale] = \
                    out[:, :, oy0:oy1, ox0:ox1]
    y = y.clamp(0, 1).float().squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_img = Image.fromarray((y * 255.0).round().astype(np.uint8), "RGB")
    print(f"[image.upscale_with_model] {image.size} -> {out_img.size} "
          f"scale={scale} tile={tile}", flush=True)
    return {"image": out_img}


def register(registry):
    registry.register(
        "upscale_model.load",
        inputs={"name": "STRING"},
        outputs={"upscale_model": "UPSCALE_MODEL"},
        description="Upscale Model Load：加载像素放大模型（spandrel 自动识别架构，"
                    "支持 RealESRGAN/SwinIR 等 .pth/.safetensors），fp16+cuda 常驻；"
                    "name 为 /models 下文件名（可不带扩展名）"
    )(upscale_model_load)
    registry.register(
        "image.upscale_with_model",
        inputs={"image": "IMAGE", "upscale_model": "UPSCALE_MODEL",
                "tile": "INT"},
        outputs={"image": "IMAGE"},
        description="Image Upscale with Model：像素空间模型放大（倍率由模型决定，"
                    "如 RealESRGAN_x4plus=4x）；tile=0 整图，>0 分块防显存溢出；"
                    "对比 image.upscale（lanczos 插值）：模型放大重建真实细节"
    )(image_upscale_with_model)
