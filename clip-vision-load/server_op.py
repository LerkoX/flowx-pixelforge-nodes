"""IPAdapter 算子（参考图驱动，对标 ComfyUI_IPAdapter_plus 三件套）。

- clip_vision.load：CLIP Vision 图像编码器（CLIP_VISION_DIR，默认
  /models/clip_vision/<目录>，diffusers 格式）
- ipadapter.load：IPAdapter 权重（IPADAPTER_DIR，默认 /models/ipadapter/<文件>，
  safetensors/bin，顶层键 image_proj + ip_adapter）
- ipadapter.apply：参考图编码成正/负两份 CLIP embeds，与权重/步窗口捆绑成
  IPABundle（MODEL 视图，可继续串 LoRA），喂 sd.sample 的 model 端口

注入机制由 app/ops.py sd.sample() 消费 bundle 实现：
- pipe.load_ip_adapter(state_dict) 装配 IPAdapterAttnProcessor2_0 +
  MultiIPAdapterImageProjection（encoder_hid_dim_type=ip_image_proj）
- 每步 unet 前向带 added_cond_kwargs={"image_embeds": [原始 CLIP embeds]}
  （CFG 拼批：cat 模式 [neg, pos]；split/area 模式正负各自前向各带各的）
- weight/步窗口：逐步改写处理器 scale（0.35.2 的处理器读自身 scale 属性，
  窗口外置 0 → 处理器跳过 IP 注意力）；采样结束 try/finally 恢复原处理器
- 编码器使用策略：apply 时临时上 GPU 编码，完事卸回 CPU（~1.3GB fp16，
  不与采样期常驻模型抢 8GB 显存）；embeds 本体很小（标准版 1024 维，
  plus 版 257×1280）驻留 GPU

标准版与 plus 版按权重 image_proj 是否含 latents 键自动判别：
- 标准版：image_embeds（pooled），负向 = zeros_like（diffusers encode_image 语义）
- plus 版：hidden_states[-2]，负向 = 零图编码（非 zeros_like！同 diffusers）

分发：本文件是节点共享算子族（单一事实源），由 nodes/_tools/sync-common.py
分发为各成员节点的 server_op.py，节点运行时经 ensure_plugin 自注册到推理服务
（POST /admin/plugins，hash 幂等）。
"""
import os

import torch
from PIL import Image

from app import ops as _core_ops  # 插件可见性约定：可复用核心原语

# 编码器缓存：name -> CLIPVisionModelWithProjection（CPU 常驻）
_VISION_CACHE = {}


def _vision_dir():
    return os.environ.get("CLIP_VISION_DIR", "/models/clip_vision")


def _ipadapter_dir():
    return os.environ.get("IPADAPTER_DIR", "/models/ipadapter")


def clip_vision_load(name):
    """CLIP Vision Load：加载 CLIP 图像编码器（diffusers 格式目录）。
    fp16 加载、CPU 常驻缓存；CLIPImageProcessor 一并挂载（apply 时复用）。"""
    base = _vision_dir()
    if os.path.basename(name) != name:
        raise ValueError(
            f"clip_vision.load: name 需为目录名，got {name!r}")
    path = os.path.join(base, name)
    if not os.path.isdir(path):
        available = sorted(os.listdir(base)) if os.path.isdir(base) else []
        raise FileNotFoundError(
            f"clip_vision '{name}' not found in {base}; "
            f"available: {available or '(empty)'}")
    enc = _VISION_CACHE.get(name)
    if enc is None:
        from transformers import (  # 懒加载：无 GPU 契约测试可导入
            CLIPImageProcessor, CLIPVisionModelWithProjection)
        enc = CLIPVisionModelWithProjection.from_pretrained(
            path, torch_dtype=torch.float16)
        enc.eval()
        enc._flowx_processor = CLIPImageProcessor.from_pretrained(path)
        _VISION_CACHE[name] = enc
        print(f"[clip_vision.load] loaded {path}", flush=True)
    return {"clip_vision": enc}


def _load_state_dict(path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file  # 懒加载
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def _unflatten_ipadapter(sd):
    """统一成嵌套结构 {image_proj: {...}, ip_adapter: {...}}。
    safetensors 只能是扁平键（'image_proj.proj.weight'），原版 .bin 是嵌套
    dict；diffusers load_ip_adapter 与 plus 判别都吃嵌套格式。"""
    if "image_proj" in sd and isinstance(sd.get("image_proj"), dict):
        return sd
    out = {}
    for k, v in sd.items():
        top, sep, rest = k.partition(".")
        if not sep:
            raise ValueError(
                f"IPAdapter 权重存在无法归类的顶层键 {k!r}（既非嵌套 dict "
                f"也非 'image_proj./ip_adapter.' 前缀扁平键）")
        out.setdefault(top, {})[rest] = v
    return out


def ipadapter_load(name):
    """IPAdapter Load：读适配器权重为 state dict（CPU，几十 MB 量级）。
    按 image_proj 是否含 latents 键判别 plus 版（apply 时决定编码路径）。"""
    base = _ipadapter_dir()
    if os.path.basename(name) != name:
        raise ValueError(
            f"ipadapter.load: name 需为纯文件名（去扩展名），got {name!r}")
    path = next((os.path.join(base, name + ext)
                 for ext in (".safetensors", ".bin", ".pt", ".pth")
                 if os.path.isfile(os.path.join(base, name + ext))), None)
    if path is None:
        available = sorted(os.listdir(base)) if os.path.isdir(base) else []
        raise FileNotFoundError(
            f"ipadapter '{name}' not found in {base}; "
            f"available: {available or '(empty)'}")
    sd = _unflatten_ipadapter(_load_state_dict(path))
    if "image_proj" not in sd or "ip_adapter" not in sd:
        raise ValueError(
            f"{path} 不是 IPAdapter 权重（缺 image_proj/ip_adapter 顶层键）")
    is_plus = "latents" in sd["image_proj"]
    print(f"[ipadapter.load] {path} ({'plus' if is_plus else 'standard'})",
          flush=True)
    return {"ipadapter": {"state_dict": sd, "name": name, "plus": is_plus}}


class IPABundle(_core_ops.ModelRef):
    """ipadapter.apply 的产物（MODEL 对象）：pipe + lora patches + ipa 注入参数。
    继承 ModelRef 使 resolve_pipe/lora_apply 链路透明可叠加。"""

    def __init__(self, pipe, patches, ipa):
        super().__init__(pipe, patches)
        # {name, state_dict, pos_embeds, neg_embeds, weight,
        #  start_percent, end_percent}
        self.ipa = ipa


def ipadapter_apply(model, ipadapter, clip_vision, image, weight=1.0,
                    start_percent=0.0, end_percent=1.0):
    """IPAdapter Apply：参考图 → CLIP embeds（正/负两份，此后采样期不再动
    编码器），与权重/步窗口捆绑成新 MODEL 视图。
    weight 常用 0.5~1.0（>1 会过拟合参考图）；start/end_percent 限定注入的
    采样步区间（风格类参考图常只注入前半段，保构图自由度）。"""
    pipe, patches = _core_ops.resolve_pipe(model)
    if not hasattr(pipe, "unet"):
        raise ValueError(
            f"ipadapter.apply: model 需为 SD1.x/SDXL 管道（带 unet），"
            f"got {type(pipe).__name__}（SD3/视频管道不支持）")
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"ipadapter.apply: image 需为单张 PIL 图像，"
            f"got {type(image).__name__}")
    if not isinstance(ipadapter, dict) or "state_dict" not in ipadapter:
        raise ValueError("ipadapter.apply: ipadapter 需为 ipadapter.load 的产物")
    weight = float(weight)
    lo, hi = float(start_percent), float(end_percent)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError(
            f"start/end_percent 需满足 0<=start<end<=1，got {lo}/{hi}")

    device = _core_ops.exec_device_of(pipe)
    enc = clip_vision
    proc = getattr(enc, "_flowx_processor", None)
    if proc is None:
        from transformers import CLIPImageProcessor  # 懒加载
        proc = CLIPImageProcessor(size=enc.config.image_size,
                                  crop_size=enc.config.image_size)
    pixels = proc(images=image.convert("RGB"), return_tensors="pt").pixel_values
    enc.to(device)
    try:
        with torch.no_grad():
            if ipadapter.get("plus"):
                pos = enc(pixels.to(device, enc.dtype),
                          output_hidden_states=True).hidden_states[-2]
                # diffusers 语义：plus 版负向是零图编码，不是 zeros_like
                neg = enc(torch.zeros_like(pixels).to(device, enc.dtype),
                          output_hidden_states=True).hidden_states[-2]
            else:
                pos = enc(pixels.to(device, enc.dtype)).image_embeds
                neg = torch.zeros_like(pos)
    finally:
        enc.to("cpu")  # 编码器卸回 CPU：~1.3GB fp16 不占采样期显存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    ipa = {"name": ipadapter.get("name", "?"),
           "state_dict": ipadapter["state_dict"],
           "pos_embeds": pos, "neg_embeds": neg,
           "weight": weight, "start_percent": lo, "end_percent": hi}
    print(f"[ipadapter.apply] '{ipa['name']}' weight={weight} "
          f"window=[{lo},{hi}) embeds={tuple(pos.shape)}", flush=True)
    return {"model": IPABundle(pipe, patches, ipa)}


def register(registry):
    registry.register(
        "clip_vision.load",
        inputs={"name": "STRING"},
        outputs={"clip_vision": "CLIP_VISION"},
        description="CLIP Vision Load：加载 CLIP 图像编码器（CLIP_VISION_DIR 下"
                    " diffusers 格式目录，如 CLIP-ViT-H-14），IPAdapter 的参考图"
                    "编码器；fp16 CPU 常驻，编码时临时上 GPU")(clip_vision_load)
    registry.register(
        "ipadapter.load",
        inputs={"name": "STRING"},
        outputs={"ipadapter": "IPADAPTER"},
        description="IPAdapter Load：读 IPADAPTER_DIR 下的适配器权重"
                    "（safetensors/bin）；标准版/plus 版自动判别")(ipadapter_load)
    registry.register(
        "ipadapter.apply",
        inputs={"model": "MODEL", "ipadapter": "IPADAPTER",
                "clip_vision": "CLIP_VISION", "image": "IMAGE",
                "weight": "FLOAT", "start_percent": "FLOAT",
                "end_percent": "FLOAT"},
        outputs={"model": "MODEL"},
        description="IPAdapter Apply：参考图驱动生成——image 编码成图像条件注入"
                    "采样（风格/内容/构图迁移）；weight 默认 1.0，"
                    "start/end_percent 限定注入步区间（默认全程）；"
                    "仅支持 SD1.x/SDXL，可与 ControlNet/LoRA 叠加")(ipadapter_apply)
