"""CatVTON 换装算子族（G11 升级：从「复用重绘」到专用 VTON 模型，dev-plan §52）。

- `vton.catvton_load`  ：装载 CatVTON（SD1.5-inpainting 基座 + CatVTON attention 权重）
                        → 返回 LRU 键字符串（**不引入新对象类型**，同 seg.sam_load 口径）
- `vton.catvton_tryon` ：person + cloth + mask → 换装结果 IMAGE（贴回原图坐标系）

为什么换模型（§52 实测）：FLUX + Redux 走的是「image prompt / 风格引导」，没有服装
结构约束，768×1024 8 步在 1080 上要 72 分钟（exec 706），且结果只能是新旧衣服**融合**。
CatVTON 的对拍机制（已读官方源码）：
- 人图 latent 与衣服 latent **沿高度维空间拼接**（`cat([masked, condition], dim=-2)`），
  UNet 输入仍是标准 inpainting 的 9 通道（4 噪声 + 1 mask + 4 masked image）；
- cross-attention 全段替换为 `SkipAttnProcessor`（**不用文本提示词**），可训练参数只有
  49.57M 的 attention ⇒ 重绘区只能从衣服 latent 取信息，这才是「搬衣服」而不是「融合」。

Pascal(sm_61) 硬约束（本族两条非默认实现，务必保留）：
1. **分块 self-attention 必须开**：绘布沿高度拼成 2 倍（768×1024 → latent 96×256 =
   24576 token），Pascal 上 SDPA 只能走 math 后端，要物化 8 头 × 24576² × 4B ≈ 19GB
   的分数矩阵 → WDDM 兜底到共享显存 → 实测 **7 分钟连第 1 步都跑不完**；按 query 分块
   （chunk=1024）后 5.4s/步（50 步 268s）。chunk=0 可关（Ampere+ 有高效内核时不必开）。
2. **默认 fp32**：Pascal 非张量 fp16 吞吐 = fp32 的 1/64（§49.2 同款判据），且 CatVTON
   官方推荐 bf16 而 sm_61 不支持 bf16。fp32 全量约 6.2GB 峰值显存（768×1024/50 步实测）。

模型口径（全部本地目录，**零网络访问**）：
- 基座：`CATVTON_BASE_DIR`（默认 `/models/stable-diffusion-inpainting`）——必须含
  `unet/` + `scheduler/`（inpainting 基座的 UNet 是 9 通道，普通 v1-5 权重不可替代）；
- attention：`CATVTON_DIR`（默认 `/models/catvton`）下 `<变体目录>/attention/model.safetensors`
- VAE：`CATVTON_VAE`（默认 `/models/vae-ft-mse-840000-ema-pruned.safetensors`，单文件）
- 蒙版：语义与 mask-paint/mask-grow 一致（白=要换掉的旧衣服区域），可直接复用现有笔画；
  AutoMasker（SCHP+DensePose，需 detectron2）**不纳入**，见 §52.5。

⚠️ 注册形态（踩过的坑，别再犯）：算子必须写成**装饰器形式**——把注册调用挂在
算子函数的装饰器位置（`@` 那一行）；写成「平铺单独一行、不挂到函数上」的话，
`Registry.register()` 返回的装饰器被丢弃：上传接口照样 200 + 归属簿照样记账
（claimed 走录制 shim）+ 静态门禁也过，但服务端注册表里**一个算子都没有**
→ 运行时 "unknown op"。本族第一版就是这么写的，e2e 才暴露（exec 707）。

分发：本文件是节点共享算子族（单一事实源），由 nodes/_tools/sync-common.py 分发为
`catvton-load` / `catvton-tryon` 的 server_op.py，节点运行时 ensure_plugin 自注册。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "34ddd651f970"
OP_FAMILY = "vton"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

import json
import os
import random
import time

import torch
from PIL import Image, ImageFilter

# 基座（inpainting UNet + DDIM scheduler；text_encoder 不用——cross-attn 被 Skip）
BASE_DIR = os.environ.get("CATVTON_BASE_DIR", "/models/stable-diffusion-inpainting")
# CatVTON attention 权重根目录（<variant_sub>/attention/model.safetensors）
ATTN_DIR = os.environ.get("CATVTON_DIR", "/models/catvton")
# VAE：单文件或 diffusers 目录都支持
VAE_PATH = os.environ.get("CATVTON_VAE",
                          "/models/vae-ft-mse-840000-ema-pruned.safetensors")

# 变体 → 权重子目录（官方 auto_attn_ckpt_load 的三档）
VARIANTS = {"mix": "mix-48k-1024",           # 1024 训练、泛化最好（默认）
            "vitonhd": "vitonhd-16k-512",
            "dresscode": "dresscode-16k-512"}
_DTYPES = {"fp32": torch.float32, "fp16": torch.float16}


# ---------------------------------------------------------------------------
# vendored：官方 model/attn_processor.py（自带 AttnProcessor2_0 副本，
# 不依赖 diffusers 内部符号 ⇒ diffusers 0.29→0.35 漂移面只剩 from_pretrained 级 API）
# ---------------------------------------------------------------------------
class SkipAttnProcessor(torch.nn.Module):
    """cross-attention 占位：直接返回输入（文本条件被整段摘除）。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        return hidden_states


class AttnProcessor2_0(torch.nn.Module):
    """官方副本（scaled_dot_product_attention 版本）。"""

    def __init__(self, hidden_size=None, cross_attention_dim=None, **kwargs):
        super().__init__()
        if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0+")

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        return _attn_2_0_call(attn, hidden_states, encoder_hidden_states,
                              attention_mask, temb)


class AttnProcessorChunked(torch.nn.Module):
    """与 AttnProcessor2_0 数学等价，但按 query 分块做 SDPA（Pascal 必需，见模块头）。

    Pascal 上 SDPA 走 math 后端会一次性物化 [heads, nq, nk] 分数矩阵；CatVTON 的
    nq=nk=24576 时约 19GB ⇒ 落到共享显存（WDDM）→ 静态退化到分钟级/步。
    """

    def __init__(self, chunk=1024, **kwargs):
        super().__init__()
        self.chunk = max(1, int(chunk))

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, *args, **kwargs):
        return _attn_2_0_call(attn, hidden_states, encoder_hidden_states,
                              attention_mask, temb, chunk=self.chunk)


def _attn_2_0_call(attn, hidden_states, encoder_hidden_states=None,
                   attention_mask=None, temb=None, chunk=0):
    """官方 AttnProcessor2_0.__call__ 主体；chunk>0 时按 query 分块。"""
    import torch.nn.functional as F

    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    input_ndim = hidden_states.ndim
    if input_ndim == 4:
        batch_size, channel, height, width = hidden_states.shape
        hidden_states = hidden_states.view(batch_size, channel,
                                           height * width).transpose(1, 2)
    batch_size, sequence_length, _ = (
        hidden_states.shape if encoder_hidden_states is None
        else encoder_hidden_states.shape)
    if attention_mask is not None:
        attention_mask = attn.prepare_attention_mask(attention_mask,
                                                     sequence_length, batch_size)
        attention_mask = attention_mask.view(batch_size, attn.heads, -1,
                                             attention_mask.shape[-1])
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    query = attn.to_q(hidden_states)
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states
    elif attn.norm_cross:
        encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)
    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    if chunk and chunk < query.shape[2]:
        out = torch.empty_like(query)
        for s in range(0, query.shape[2], chunk):
            e = min(s + chunk, query.shape[2])
            out[:, :, s:e] = F.scaled_dot_product_attention(
                query[:, :, s:e], key, value, attn_mask=attention_mask,
                dropout_p=0.0, is_causal=False)
        hidden_states = out
    else:
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0,
            is_causal=False)
    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim)
    hidden_states = hidden_states.to(query.dtype)
    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)
    if input_ndim == 4:
        hidden_states = hidden_states.transpose(-1, -2).reshape(
            batch_size, channel, height, width)
    if attn.residual_connection:
        hidden_states = hidden_states + residual
    return hidden_states / attn.rescale_output_factor


def _init_adapter(unet, chunk, cross_attn_cls=SkipAttnProcessor):
    """vendored 官方 model/utils.py::init_adapter（self-attn 走分块版）。"""
    cross_attn_dim = unet.config.cross_attention_dim
    self_attn_cls = AttnProcessorChunked if int(chunk) > 0 else AttnProcessor2_0
    attn_procs = {}
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else cross_attn_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = self_attn_cls(hidden_size=hidden_size,
                                             cross_attention_dim=cross_attention_dim,
                                             chunk=int(chunk))
        else:
            attn_procs[name] = cross_attn_cls(hidden_size=hidden_size,
                                              cross_attention_dim=cross_attention_dim)
    unet.set_attn_processor(attn_procs)
    return torch.nn.ModuleList(unet.attn_processors.values())


def _trainable_attention(unet):
    """vendored 官方 get_trainable_module(unet, "attention")：所有 attn1 块。"""
    blocks = torch.nn.ModuleList()
    for name, mod in unet.named_modules():
        if "attn1" in name:
            blocks.append(mod)
    return blocks


# ---------------------------------------------------------------------------
# vendored：官方 utils.py（图片预处理 / VAE 编码 / 贴回）
# ---------------------------------------------------------------------------
def _compute_vae_encodings(image, vae):
    pixel_values = image.to(memory_format=torch.contiguous_format).float()
    pixel_values = pixel_values.to(vae.device, dtype=vae.dtype)
    with torch.no_grad():
        model_input = vae.encode(pixel_values).latent_dist.sample()
    return model_input * vae.config.scaling_factor


def _prepare_image(image):
    """PIL/tensor → [-1,1] NCHW float32 tensor。"""
    import numpy as np
    if isinstance(image, torch.Tensor):
        if image.ndim == 3:
            image = image.unsqueeze(0)
        return image.to(dtype=torch.float32)
    if isinstance(image, (Image.Image, np.ndarray)):
        image = [image]
    if isinstance(image, list) and isinstance(image[0], Image.Image):
        image = np.concatenate([np.array(i.convert("RGB"))[None, :] for i in image],
                               axis=0)
    elif isinstance(image, list) and isinstance(image[0], np.ndarray):
        image = np.concatenate([i[None, :] for i in image], axis=0)
    image = image.transpose(0, 3, 1, 2)
    return torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0


def _prepare_mask_image(mask_image):
    """PIL/tensor → {0,1} N1HW tensor。"""
    import numpy as np
    if isinstance(mask_image, torch.Tensor):
        if mask_image.ndim == 2:
            mask_image = mask_image.unsqueeze(0).unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] == 1:
            mask_image = mask_image.unsqueeze(0)
        elif mask_image.ndim == 3 and mask_image.shape[0] != 1:
            mask_image = mask_image.unsqueeze(1)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
    else:
        if isinstance(mask_image, (Image.Image, np.ndarray)):
            mask_image = [mask_image]
        if isinstance(mask_image, list) and isinstance(mask_image[0], Image.Image):
            mask_image = np.concatenate(
                [np.array(m.convert("L"))[None, None, :] for m in mask_image], axis=0)
            mask_image = mask_image.astype(np.float32) / 255.0
        elif isinstance(mask_image, list) and isinstance(mask_image[0], np.ndarray):
            mask_image = np.concatenate([m[None, None, :] for m in mask_image], axis=0)
        mask_image[mask_image < 0.5] = 0
        mask_image[mask_image >= 0.5] = 1
        mask_image = torch.from_numpy(mask_image)
    return mask_image


def _crop_box(size, target):
    """resize_and_crop 的裁框（与官方同口径）：返回 (左, 上, 右, 下)。"""
    w, h = size
    tw, th = target
    if w / h < tw / th:
        new_w, new_h = w, w * th // tw
    else:
        new_h, new_w = h, h * tw // th
    return ((w - new_w) // 2, (h - new_h) // 2,
            (w - new_w) // 2 + new_w, (h - new_h) // 2 + new_h)


def _resize_crop(image, size):
    box = _crop_box(image.size, size)
    return image.crop(box).resize(size, Image.LANCZOS)


def _resize_pad(image, size):
    """衣服图按比例缩放 + 白底补边（官方 resize_and_padding 口径）。"""
    w, h = image.size
    tw, th = size
    if w / h < tw / th:
        new_h, new_w = th, w * th // h
    else:
        new_w, new_h = tw, h * tw // w
    image = image.resize((new_w, new_h), Image.LANCZOS)
    pad = Image.new("RGB", size, (255, 255, 255))
    pad.paste(image, ((tw - new_w) // 2, (th - new_h) // 2))
    return pad


def _numpy_to_pil(images):
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        return [Image.fromarray(image.squeeze(), mode="L") for image in images]
    return [Image.fromarray(image) for image in images]


# ---------------------------------------------------------------------------
# 管道句柄 + LRU（键 "catvton:<variant>:<dtype>:<vae_dtype>:<chunk>"，
# 进 model_manager LRU ⇒ /models 常驻列表、model-unload、淘汰/pin 全套语义）
# ---------------------------------------------------------------------------
class CatVTONHandle:
    """已装载的 CatVTON：unet/vae/scheduler + 元信息（dtype/变体/分块）。"""

    def __init__(self, unet, vae, scheduler, variant, dtype_name,
                 vae_dtype_name, chunk, load_seconds):
        self.unet = unet
        self.vae = vae
        self.scheduler = scheduler
        self.variant = variant
        self.dtype_name = dtype_name
        self.vae_dtype_name = vae_dtype_name
        self.chunk = chunk
        self.load_seconds = load_seconds

    def info(self):
        p = sum(x.numel() for x in self.unet.parameters())
        return (f"variant={self.variant} unet={self.dtype_name} "
                f"vae={self.vae_dtype_name} chunk={self.chunk} "
                f"params={p/1e6:.0f}M load={self.load_seconds:.0f}s")


def _models():
    from app.main import models
    return models


def _register(key, obj, est_bytes, arch):
    models = _models()
    with models._lock:
        models._evict_if_needed(est_bytes)
        models._pipes[key] = obj
        models._archs[key] = arch
        models._sizes[key] = est_bytes
        models._last_used[key] = time.time()
    print(f"[vton] registered '{key}' est={est_bytes/1e9:.2f}GB", flush=True)
    return obj


def _cached(key):
    models = _models()
    with models._lock:
        return models._pipes.get(key)


def _load_vae(path, dtype):
    from diffusers import AutoencoderKL
    if os.path.isfile(path):
        vae = AutoencoderKL.from_single_file(path)
    elif os.path.isfile(os.path.join(path, "config.json")):
        vae = AutoencoderKL.from_pretrained(path)
    else:
        raise FileNotFoundError(
            f"vton: VAE 路径 {path} 既不是单文件也不是 diffusers 目录"
            f"（环境变量 CATVTON_VAE）")
    return vae.to(dtype)


def _handle(variant, dtype_name, vae_dtype_name, chunk, op_name):
    """LRU 命中直接复用；否则装载（约 160s，主要耗时在 3.4GB 基座反序列化）。"""
    if variant not in VARIANTS:
        raise ValueError(f"{op_name}: variant 非法 {variant!r}，可选 "
                         f"{sorted(VARIANTS)}")
    if dtype_name not in _DTYPES or vae_dtype_name not in _DTYPES:
        raise ValueError(f"{op_name}: dtype/vae_dtype 只能是 {sorted(_DTYPES)}")
    chunk = int(chunk or 0)
    key = f"catvton:{variant}:{dtype_name}:{vae_dtype_name}:{chunk}"
    hit = _cached(key)
    if hit is not None:
        return key, hit

    if not torch.cuda.is_available():
        raise RuntimeError(f"{op_name}: 需要 CUDA（CatVTON 为 GPU 推理）")
    unet_dir = os.path.join(BASE_DIR, "unet")
    if not os.path.isdir(unet_dir):
        raise FileNotFoundError(
            f"{op_name}: 基座缺 unet/ 目录：{unet_dir}（环境变量 CATVTON_BASE_DIR="
            f"{BASE_DIR}；必须是 SD1.5 **inpainting** 基座，普通 v1-5 的 UNet 是 4 通道）")
    sub = VARIANTS[variant]
    attn_path = os.path.join(ATTN_DIR, sub, "attention")
    if not os.path.isdir(attn_path):
        raise FileNotFoundError(
            f"{op_name}: 缺 attention 权重目录 {attn_path}（环境变量 CATVTON_DIR="
            f"{ATTN_DIR}；需 <变体>/{sub}/attention/model.safetensors）")

    from accelerate import load_checkpoint_in_model
    from diffusers import DDIMScheduler, UNet2DConditionModel

    t0 = time.time()
    dtype = _DTYPES[dtype_name]
    vae_dtype = _DTYPES[vae_dtype_name]
    scheduler = DDIMScheduler.from_pretrained(BASE_DIR, subfolder="scheduler")
    vae = _load_vae(VAE_PATH, vae_dtype).to("cuda")
    print(f"[vton] vae loaded ({vae_dtype_name}) {time.time()-t0:.1f}s", flush=True)
    unet = UNet2DConditionModel.from_pretrained(BASE_DIR, subfolder="unet")
    unet = unet.to("cuda")
    attn_modules = _init_adapter(unet, chunk)
    load_checkpoint_in_model(attn_modules, attn_path)
    unet = unet.to(dtype)
    unet.eval()
    vae.eval()
    handle = CatVTONHandle(unet, vae, scheduler, variant, dtype_name,
                           vae_dtype_name, chunk, time.time() - t0)
    est = sum(p.numel() * p.element_size() for p in unet.parameters())
    est += sum(p.numel() * p.element_size() for p in vae.parameters())
    _register(key, handle, est, f"CatVTON({variant}, {dtype_name}, chunk={chunk})")
    print(f"[vton] {op_name}: {handle.info()}", flush=True)
    return key, handle


# ---------------------------------------------------------------------------
# 采样
# ---------------------------------------------------------------------------
def _denoise(handle, person, cloth, mask, steps, guidance, seed, width, height,
             job, rec, preview_every, log):
    """vendored 官方 model/pipeline.py::CatVTONPipeline.__call__（去 safety_checker）。"""
    from diffusers.utils.torch_utils import randn_tensor

    concat_dim = -2  # 官方注释保留：latent 画布沿高度拼接
    person_t = _prepare_image(person).to("cuda", dtype=handle.unet.dtype)
    cloth_t = _prepare_image(cloth).to("cuda", dtype=handle.unet.dtype)
    mask_t = _prepare_mask_image(mask).to("cuda", dtype=handle.unet.dtype)
    masked_image = person_t * (mask_t < 0.5)
    masked_latent = _compute_vae_encodings(masked_image, handle.vae)
    condition_latent = _compute_vae_encodings(cloth_t, handle.vae)
    mask_latent = torch.nn.functional.interpolate(
        mask_t, size=masked_latent.shape[-2:], mode="nearest")
    del person_t, cloth_t, mask_t, masked_image
    masked_latent_concat = torch.cat([masked_latent, condition_latent], dim=concat_dim)
    mask_latent_concat = torch.cat([mask_latent, torch.zeros_like(mask_latent)],
                                   dim=concat_dim)
    gen = torch.Generator("cpu").manual_seed(int(seed))
    latents = randn_tensor(masked_latent_concat.shape, generator=gen,
                           device=masked_latent_concat.device,
                           dtype=handle.unet.dtype)
    handle.scheduler.set_timesteps(int(steps), device="cuda")
    timesteps = handle.scheduler.timesteps
    latents = latents * handle.scheduler.init_noise_sigma
    cfg_on = float(guidance) > 1.0
    if cfg_on:
        masked_latent_concat = torch.cat([
            torch.cat([masked_latent, torch.zeros_like(condition_latent)],
                      dim=concat_dim),
            masked_latent_concat])
        mask_latent_concat = torch.cat([mask_latent_concat] * 2)

    from app import execution
    total = len(timesteps)
    t0 = time.time()
    step_eta = None
    for i, t in enumerate(timesteps):
        execution.check_cancelled()
        non_inpaint = torch.cat([latents] * 2) if cfg_on else latents
        non_inpaint = handle.scheduler.scale_model_input(non_inpaint, t)
        model_in = torch.cat([non_inpaint, mask_latent_concat,
                              masked_latent_concat], dim=1)
        with torch.no_grad():
            noise_pred = handle.unet(model_in, t.to("cuda"),
                                     encoder_hidden_states=None, return_dict=False)[0]
        if cfg_on:
            uncond, cond = noise_pred.chunk(2)
            noise_pred = uncond + float(guidance) * (cond - uncond)
        kw = {}
        import inspect
        sig = inspect.signature(handle.scheduler.step).parameters
        if "eta" in sig:
            kw["eta"] = 1.0          # 官方默认 eta=1.0（DDIM 随机性口径）
        if "generator" in sig:
            kw["generator"] = gen
        latents = handle.scheduler.step(noise_pred, t, latents, **kw).prev_sample
        del noise_pred, model_in
        done = i + 1
        if job:
            job.set_progress(done, total)
        if step_eta is None and done == 1:
            step_eta = time.time() - t0
        if rec and rec.want(i, total):
            try:
                rec.push_pil(_latent_preview(handle, latents), done / total)
            except Exception:
                from app import preview
                rec.push_pil(preview.progress_card(done, total), done / total)
        if done == 1 or done % 5 == 0 or done == total:
            log(f"[vton] step {done}/{total} t={float(t):.1f} "
                f"elapsed={time.time()-t0:.1f}s "
                f"vram={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    latents = latents.split(latents.shape[concat_dim] // 2, dim=concat_dim)[0]
    latents = latents.to("cuda", dtype=handle.vae.dtype) / handle.vae.config.scaling_factor
    with torch.no_grad():
        out = handle.vae.decode(latents).sample
    out = (out / 2 + 0.5).clamp(0, 1)
    out = out.cpu().permute(0, 2, 3, 1).float().numpy()
    return _numpy_to_pil(out)[0], time.time() - t0


def _latent_preview(handle, latents, max_size=256):
    """过程帧：解码半分辨率 latent 的上半幅（人图那半）→ 缩到 ≤256px。"""
    half = latents.split(latents.shape[-2] // 2, dim=-2)[0]
    with torch.no_grad():
        img = handle.vae.decode(
            half.to(dtype=handle.vae.dtype) / handle.vae.config.scaling_factor).sample
    img = ((img / 2 + 0.5).clamp(0, 1)[0].cpu().permute(1, 2, 0).float().numpy() * 255)
    pil = Image.fromarray(img.round().astype("uint8"))
    if max(pil.size) > max_size:
        pil.thumbnail((max_size, max_size), Image.LANCZOS)
    return pil


def _fit_back(person, result, mask, repaint, width, height):
    """把 768×1024 画布结果贴回**原图坐标系**（§52.6：脸/背景按 mask 保留）。

    官方 resize_and_crop 会把人图按目标比例居中裁切，所以结果只覆盖原图中间那块；
    这里按同一裁框把结果缩回原尺寸贴回，裁掉的上下/左右边条保留原像素。
    """
    box = _crop_box(person.size, (width, height))
    box_size = (box[2] - box[0], box[3] - box[1])
    resized = result.resize(box_size, Image.LANCZOS)
    if not repaint:
        out = person.copy()
        out.paste(resized, (box[0], box[1]))
        return out
    mask_crop = mask.crop(box).resize(box_size, Image.LANCZOS)
    patch = Image.composite(resized, person.crop(box), mask_crop)
    out = person.copy()
    out.paste(patch, (box[0], box[1]))
    return out


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------
def catvton_load(variant="mix", dtype="fp32", vae_dtype="fp32", chunk=1024):
    """装载 CatVTON → LRU 键（供 catvton_tryon 引用；可被 model-unload 卸载）。"""
    t0 = time.time()
    key, handle = _handle(variant, dtype, vae_dtype, chunk, "vton.catvton_load")
    print(f"[vton] catvton_load -> {key} ({time.time()-t0:.1f}s)", flush=True)
    return {"vton": key, "info": handle.info()}


def catvton_tryon(vton, person, cloth, mask, steps=50, guidance=2.5, seed=-1,
                  width=768, height=1024, mask_blur=9, repaint=True,
                  preview_every=5):
    """CatVTON 换装：person + cloth + mask → 结果 IMAGE（贴回原图坐标系）。"""
    key = vton if isinstance(vton, str) else str(vton)
    handle = _cached(key)
    if handle is None:
        raise RuntimeError(
            f"vton.catvton_tryon: 管道句柄 '{key}' 不在常驻表里（可能被 model-unload "
            f"淘汰或服务重启）——请重跑上游 catvton-load 节点")
    # 本族用 STRING 键传句柄（不引入新对象类型），引擎的对象身份 pin 认不出来 ⇒
    # 这里显式 pin，防采样期间被别人的 LRU 淘汰（finally 里 unpin）。
    models = _models()
    with models._lock:
        models._last_used[key] = time.time()
    models.pin(key)
    try:
        return _tryon_pinned(key, handle, person, cloth, mask, steps, guidance,
                             seed, width, height, mask_blur, repaint,
                             preview_every)
    finally:
        models.unpin(key)


def _tryon_pinned(key, handle, person, cloth, mask, steps, guidance, seed,
                  width, height, mask_blur, repaint, preview_every):
    """catvton_tryon 的实际执行体（句柄已 pin）。"""
    from app import execution, preview

    width, height = int(width), int(height)
    steps = int(steps)
    seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**32 - 1)
    person = person if person.mode == "RGB" else person.convert("RGB")
    cloth = cloth if cloth.mode == "RGB" else cloth.convert("RGB")
    mask = mask.convert("L")
    if mask.size != person.size:
        mask = mask.resize(person.size, Image.LANCZOS)
    if int(mask_blur) > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(int(mask_blur)))

    person_c = _resize_crop(person, (width, height))
    cloth_c = _resize_pad(cloth, (width, height))
    mask_c = _resize_crop(mask, (width, height))

    job = execution.current()
    rec = (preview.recorder_for(job.id, int(preview_every))
           if job and int(preview_every) > 0 else None)
    if rec:
        try:
            dbg = person_c.copy()
            dbg.paste(Image.new("RGB", (width // 4, height // 4), (34, 211, 238)),
                      (8, 8))
            rec.push_pil(_thumb(dbg, 256), 0.0)
        except Exception:
            pass

    print(f"[vton] catvton_tryon key={key} {width}x{height} steps={steps} "
          f"cfg={guidance} seed={seed} blur={mask_blur} repaint={repaint} "
          f"mask_coverage={_coverage(mask_c):.3f}", flush=True)
    coverage = _coverage(mask_c)
    result, seconds = _denoise(handle, person_c, cloth_c, mask_c, steps, guidance,
                               seed, width, height, job, rec, preview_every,
                               print)
    out = _fit_back(person, result, mask, bool(repaint), width, height)
    peak = torch.cuda.max_memory_allocated() / 1e9
    info = json.dumps({"key": key, "variant": handle.variant, "dtype": handle.dtype_name,
                       "chunk": handle.chunk, "width": width, "height": height,
                       "steps": steps, "guidance": float(guidance), "seed": seed,
                       "mask_blur": int(mask_blur), "repaint": bool(repaint),
                       "mask_coverage": round(coverage, 4),
                       "seconds": round(seconds, 1),
                       "peak_vram_gb": round(peak, 2)}, ensure_ascii=False)
    print(f"[vton] catvton_tryon done in {seconds:.1f}s "
          f"peak_vram={peak:.2f}GB -> {out.size[0]}x{out.size[1]}", flush=True)
    return {"image": out, "info": info}


def _thumb(pil, max_size):
    out = pil.copy()
    out.thumbnail((max_size, max_size), Image.LANCZOS)
    return out


def _coverage(mask):
    hist = mask.histogram()
    total = sum(hist) or 1
    white = sum(hist[128:])
    return white / total


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "vton.catvton_load",
        inputs={"variant": "STRING", "dtype": "STRING", "vae_dtype": "STRING",
                "chunk": "INT"},
        outputs={"vton": "STRING", "info": "STRING"},
        description="装载 CatVTON 换装管道（SD1.5-inpainting 基座 + CatVTON attention，"
                    "变体 mix/vitonhd/dresscode，默认 fp32 + chunk=1024）→ 常驻 LRU 键，"
                    "喂 vton.catvton_tryon。首次装载 ~160s，之后命中直接复用")
    def _catvton_load(variant="mix", dtype="fp32", vae_dtype="fp32", chunk=1024):
        return catvton_load(variant=variant, dtype=dtype, vae_dtype=vae_dtype,
                            chunk=chunk)

    @registry.register(
        "vton.catvton_tryon",
        inputs={"vton": "STRING", "person": "IMAGE", "cloth": "IMAGE",
                "mask": "IMAGE", "steps": "INT", "guidance": "FLOAT",
                "seed": "INT", "width": "INT", "height": "INT",
                "mask_blur": "INT", "repaint": "BOOL", "preview_every": "INT"},
        outputs={"image": "IMAGE", "info": "STRING"},
        description="CatVTON 真换装：person + cloth + mask（白=要换掉的旧衣服区）→ "
                    "换装结果图（按原图坐标系贴回，repaint 时蒙版外保留原像素）。"
                    "768×1024/50 步在 GTX1080 fp32 约 270s")
    def _catvton_tryon(vton, person, cloth, mask, steps=50, guidance=2.5, seed=-1,
                       width=768, height=1024, mask_blur=9, repaint=True,
                       preview_every=5):
        return catvton_tryon(vton=vton, person=person, cloth=cloth, mask=mask,
                             steps=steps, guidance=guidance, seed=seed,
                             width=width, height=height, mask_blur=mask_blur,
                             repaint=repaint, preview_every=preview_every)
