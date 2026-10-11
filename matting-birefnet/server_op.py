"""BiRefNet 抠图算子族（G1 / B2 批次，dev-plan §四十八）。

- matting.birefnet：IMAGE → 前景 alpha 蒙版（灰度 IMAGE）+ RGBA 去背图

模型：BiRefNet（ZhengPeng7），经 transformers `AutoModelForImageSegmentation` +
`trust_remote_code=True` 加载——变体目录里必须同时有自定义建模代码
（`birefnet.py` / `BiRefNet_config.py`）与权重 `model.safetensors`，
`config.json` 的 `bb_pretrained=false` 表示不再拉 backbone 预训练权重。
变体：lite（177MB，默认）/ main（444MB）/ portrait（885MB，人像发丝）。
目录：`MATTING_MODELS_DIR`（默认 `/models/matting/<variant>/`），**零网络访问**；
缺文件时报错直接点名缺哪个、该放哪（清单见 nodes/README.md）。

口径（与既有蒙版节点一致，不引入新对象类型）：
- mask = 灰度 IMAGE（0=背景、255=前景），可直接喂 mask-grow / mask-feather /
  image-composite / latent-set-noise-mask；
- image = RGBA 去背图（PNG 带 alpha），便于直接落盘/预览；
- `refine=True` 走官方 `refine_foreground`（Photoroom 前后景分离的 blur-fusion 估计器，
  cv2 实现）修正边缘白边与彩色溢出——这是 BiRefNet 出图质量的关键一步；
- fp16 + cuda 推理；`use_checkpoint=True` 给模型内可用的块打开梯度检查点（8GB 卡兜底）。

分发：本文件是节点共享算子族（单一事实源），由 nodes/_tools/sync-common.py
分发为各成员节点的 server_op.py，节点运行时经 ensure_plugin 自注册。
依赖：transformers / timm / kornia / einops / cv2(headless) / numpy——一律函数内懒加载。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "005509206da7"
OP_FAMILY = "matting"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


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

from PIL import Image

MATTING_DIR = os.environ.get("MATTING_MODELS_DIR", "/models/matting")

# 变体 → 目录名（同时也是 MATTING_MODELS_DIR 下的子目录名）
VARIANTS = ("lite", "main", "portrait")

# 变体目录必备文件：前三个是 trust_remote_code 的自定义建模代码，必须与权重同目录
REQUIRED_FILES = ("config.json", "birefnet.py", "BiRefNet_config.py",
                  "model.safetensors")

_MODELS = {}  # (variant, use_checkpoint) -> torch.nn.Module（进程级常驻）

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _variant_dir(variant, op_name):
    """变体目录定位 + 必备文件检查（纯 os 操作，先快速失败）。"""
    if variant not in VARIANTS:
        raise ValueError(
            f"{op_name}: variant 非法 {variant!r}，可选 {VARIANTS}"
            f"（HR/matting/dynamic 变体未纳入本批，见 dev-plan §48.7）")
    path = os.path.join(MATTING_DIR, variant)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{op_name}: 变体目录不存在 {path}"
            f"（下载见 nodes/README.md 权重清单；环境变量 MATTING_MODELS_DIR="
            f"{MATTING_DIR}）")
    missing = [f for f in REQUIRED_FILES
               if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(
            f"{op_name}: 变体 {variant} 权重/建模代码缺失：{path}/{'、'.join(missing)}"
            f"（四个文件必须同目录：config.json / birefnet.py / BiRefNet_config.py / "
            f"model.safetensors）")
    return path


def _use_checkpoint_flag(model, op_name):
    """给模型内声明了 use_checkpoint 的块打开梯度检查点；返回被打开的块数。

    BiRefNet 的 use_checkpoint 是各 Block 的构造参数（不是 config 字段），
    官方权重里默认 False——这里统一巡一遍模块把可用的打开，省显存给 8GB 卡兜底。
    """
    n = 0
    for mod in model.modules():
        if hasattr(mod, "use_checkpoint") and not getattr(mod, "use_checkpoint"):
            try:
                mod.use_checkpoint = True
                n += 1
            except Exception:  # 只读属性等：忽略
                pass
    return n


def _load(variant, use_checkpoint, op_name):
    """变体模型（进程级单例）：fp16 + cuda + eval。"""
    key = (variant, bool(use_checkpoint))
    if key in _MODELS:
        return _MODELS[key]
    path = _variant_dir(variant, op_name)  # 先快速失败（FileNotFoundError）
    import torch  # 懒加载
    from transformers import AutoModelForImageSegmentation
    if not torch.cuda.is_available():
        raise RuntimeError(f"{op_name}: 需要 CUDA（BiRefNet 为 GPU 推理），"
                           f"当前 torch.cuda.is_available()=False")
    model = AutoModelForImageSegmentation.from_pretrained(
        path, trust_remote_code=True, local_files_only=True)
    model.eval().to("cuda")
    model.half()  # 官方 handler 同款：fp16 推理
    extra = ""
    if use_checkpoint:
        extra = f"，use_checkpoint 已打开 {_use_checkpoint_flag(model, op_name)} 个块"
    _MODELS[key] = model
    print(f"[{op_name}] BiRefNet 已加载：{path}（fp16/cuda{extra}）", flush=True)
    return model


def _to_tensor(img, resolution):
    """PIL → (1,3,R,R) float32 fp16-ready，ImageNet 归一化（等价官方 transforms 组合）。"""
    import numpy as np
    import torch
    x = img.convert("RGB").resize((resolution, resolution), Image.BILINEAR)
    arr = np.asarray(x, dtype=np.float32) / 255.0
    arr = (arr - np.asarray(_IMAGENET_MEAN, dtype=np.float32)) / \
        np.asarray(_IMAGENET_STD, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def _refine_foreground(image, mask_img, r=90):
    """官方 refine_foreground：blur-fusion 前后景分离（去边缘白边/彩色溢出）。

    来源：Photoroom fast-foreground-estimation，经 BiRefNet 官方 handler 引用；
    纯 cv2（容器内为 headless 版）+ numpy 实现，无额外依赖。
    """
    import cv2
    import numpy as np

    def _est(F, B, alpha):
        blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]
        blurred_FA = cv2.blur(F * alpha, (r, r))
        blurred_F = blurred_FA / (blurred_alpha + 1e-5)
        blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
        blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
        out = blurred_F + alpha * (image_arr - alpha * blurred_F
                                   - (1 - alpha) * blurred_B)
        return np.clip(out, 0, 1), blurred_B

    image_arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    alpha = np.asarray(mask_img, dtype=np.float32) / 255.0
    if alpha.shape[:2] != image_arr.shape[:2]:
        alpha = np.asarray(Image.fromarray((alpha * 255).astype(np.uint8))
                           .resize(image.size), dtype=np.float32) / 255.0
    alpha = alpha[:, :, None]
    F, blur_B = _est(image_arr, image_arr, alpha)
    F, _ = _est(F, blur_B, alpha)
    return Image.fromarray((F * 255.0).round().astype(np.uint8), "RGB")


def matting_birefnet(image, variant="lite", resolution=1024, use_checkpoint=False,
                     refine=True, refine_radius=90, alpha_gamma=1.0, invert=False):
    """BiRefNet 抠图：IMAGE → mask（灰度 alpha）+ image（RGBA 去背图）。

    - variant：lite / main / portrait
    - resolution：推理分辨率（1024/1536/2048；越大越细越慢越吃显存）
    - refine：是否做前后景分离修正（去白边，官方口径，默认开）
    - alpha_gamma：alpha 曲线（>1 前景更饱满，<1 更克制），1.0 = 不改
    - invert：蒙版反相（255-alpha，用于抠背景而非主体）
    """
    op_name = f"matting.birefnet@{OP_VERSION}"
    if not isinstance(image, Image.Image):
        raise ValueError(f"{op_name}: image 需为 IMAGE 对象，got "
                         f"{type(image).__name__}")
    resolution = int(resolution)
    if resolution < 256 or resolution > 4096:
        raise ValueError(f"{op_name}: resolution 需在 256~4096，got {resolution}")

    # 先做权重/目录检查再碰 torch：缺权重时的报错必须是"缺哪个文件"，
    # 而不是 CUDA/依赖报错（本地无 GPU 也能测这段契约）。
    model = _load(variant, use_checkpoint, op_name)
    import numpy as np
    import torch

    src = image.convert("RGB")
    x = _to_tensor(src, resolution).to("cuda").half()
    with torch.no_grad():
        preds = model(x)[-1].sigmoid().float().cpu()
    alpha = preds[0].squeeze().numpy()  # (R, R) ∈ [0,1]

    # alpha 缩放回输入尺寸 + gamma 曲线（先 8bit 再曲线，保证与节点侧口径一致）
    a8 = Image.fromarray((alpha * 255.0).round().astype(np.uint8), "L")
    if a8.size != src.size:
        a8 = a8.resize(src.size, Image.BILINEAR)
    if float(alpha_gamma) != 1.0:
        g = max(1e-3, float(alpha_gamma))
        lut = [min(255, max(0, round(255.0 * ((i / 255.0) ** (1.0 / g)))))
               for i in range(256)]
        a8 = a8.point(lut)
    if invert:
        a8 = Image.fromarray(255 - np.asarray(a8, dtype=np.uint8), "L")

    # 去背图：refine 走前后景分离（RGB 也被修正），否则直接用原图 + alpha
    rgb = _refine_foreground(src, a8, int(refine_radius)) if refine else src
    rgba = rgb.convert("RGBA")
    rgba.putalpha(a8)

    arr = np.asarray(a8)
    fg_ratio = float((arr > 127).mean())
    print(f"[{op_name}] {src.size} variant={variant} res={resolution} "
          f"refine={bool(refine)} 前景占比={fg_ratio:.3f}", flush=True)
    return {"mask": a8, "image": rgba, "width": src.size[0], "height": src.size[1]}


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "matting.birefnet",
        inputs={"image": "IMAGE", "variant": "STRING", "resolution": "INT",
                "use_checkpoint": "BOOL", "refine": "BOOL",
                "refine_radius": "INT", "alpha_gamma": "FLOAT",
                "invert": "BOOL"},
        outputs={"mask": "IMAGE", "image": "IMAGE", "width": "INT",
                 "height": "INT"},
        description="BiRefNet 抠图（G1）：IMAGE → 前景 alpha 蒙版（灰度 IMAGE）"
                    "+ RGBA 去背图。variant=lite|main|portrait，resolution 1024 起，"
                    "refine=去白边的前后景分离修正；权重在 MATTING_MODELS_DIR"
                    "（默认 /models/matting/<variant>/）"
    )(matting_birefnet)
