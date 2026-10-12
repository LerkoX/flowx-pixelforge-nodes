"""IC-Light 重打光算子族（G9 / B4 批次，dev-plan §五十 50.3 S3）。

算子：
- `iclight.load`    ：装载 IC-Light 变体（SD15 基座 + delta 权重）→ 模型键 `iclight:<variant>`
- `iclight.relight` ：前景（可带蒙版自动合成 127 灰底）+ 光方向/背景 → 重打光图

**实现来源（对照上游）**：https://github.com/lllyasviel/IC-Light （2026-10-12 取 main）
- `gradio_demo.py`（fc 变体）/ `gradio_demo_bg.py`（fbc 变体）的推理核
- 官方三处关键机制**逐行等价**保留：
  1. UNet `conv_in` 扩通道手术：`weight.zero_()` → `weight[:, :4] = 原 conv_in` → 复用原 bias
  2. IC-Light 权重是**增量（delta）**：`merged = origin + offset`，`load_state_dict(strict=True)`
  3. 条件注入：猴子补丁 `unet.forward`，从 `cross_attention_kwargs['concat_conds']` 取
     `[fg_latent]`（fc，8 通道）或 `[fg_latent, bg_latent]`（fbc，12 通道）拼到 sample 通道维
- 与本服务不同之处（有意）：
  - 官方用 `briarmbg/BriaRMBG` 抠背景；本族**复用 B2 的 BiRefNet 抠图**（matting-birefnet），
    或由调用方直接给已抠好的前景 + `mask`（本族按官方 `run_rmbg(sigma=0)` 的公式
    `127 + (img - 127) * alpha` 合成 127 灰底）。
  - 官方默认 fp16/bfloat16；本族默认 **fp32**（Pascal 纪律：GP104 非张量 fp16 吞吐 = fp32 的
    1/64，dev-plan §49.2；`dtype=auto` 在算力 < 7.0 取 fp32）。
  - 官方 delta 用 fp16 与原权重相加；本族把 delta 提升到基座精度后相加（fp32 累加更稳）。

**兼容性实测（spike，dev-plan §50.10）**：diffusers **0.35.2** + transformers 4.44.2 下
键匹配 686/686（strict）、双条件注入正常、光方向物理翻转（left Δ=+62.0 / right Δ=−89.6）、
同种子**逐字节可复现**（DDIM 与 dpmpp-sde 两遍差 0）。

依赖：torch + diffusers + safetensors + numpy（全部已在推理镜像内）
⇒ **零新增依赖、不重建推理镜像**。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
OP_VERSION = "607a43c54345"
OP_FAMILY = "iclight"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


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
import time

from PIL import Image

ICLIGHT_DIR = os.environ.get("ICLIGHT_MODELS_DIR", "/models/iclight")
# 基座：必须是 **diffusers 目录格式**的 SD1.5（unet/vae/text_encoder/tokenizer），
# 不能是单文件 .safetensors（IC-Light 的 delta 是针对这个结构做的）。
BASE_DIR = os.environ.get("ICLIGHT_BASE_DIR", "/models/v1-5-pruned-emaonly-fp16")

# 变体表：conv_in 通道数 = 4（latent）+ 4*(前景/背景条件数)
VARIANTS = {
    "fc": {"file": "iclight_sd15_fc.safetensors", "cond": 1, "conv_in": 8,
           "grad": (0, 255),
           "desc": "前景条件版（1.64GB，默认）：光方向由 light_dir + prompt 决定"},
    "fbc": {"file": "iclight_sd15_fbc.safetensors", "cond": 2, "conv_in": 12,
            "grad": (32, 224),
            "desc": "前景+背景条件版（1.64GB）：给 bg 图则主体按该背景打光"},
}

# 光方向 → 初始背景/条件（fc 用作 i2i 初始图；fbc 用作背景条件）
LIGHT_DIRS = ("none", "left", "right", "top", "bottom")

# ⚠️ 不在模块级缓存 Bundle：那样会强引用住已被 LRU 淘汰的模型，显存永不释放
#（B4/S3 真机实测踩到：淘汰 iclight:fc 后 free 仍只剩 176MB，fbc 推理慢 40 倍）。
# Bundle 只放在 model_manager 的 _pipes 里，取用一律 _cached() 现查。
_DTYPES = {}    # 模型键 -> dtype 字符串（纯元数据，不持有模型）
_LOADINFO = {}  # 模型键 -> 装载诊断


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _dtype(name, op_name):
    """`auto` 按算力选（<7.0 用 fp32，Pascal 非张量 fp16 吞吐 1/64）。"""
    import torch
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        cc = cap[0] + cap[1] / 10.0
    else:
        cc = 0.0
    n = str(name or "auto").strip().lower()
    if n in ("", "auto"):
        n = "fp32" if cc < 7.0 else "fp16"
    if n not in ("fp32", "fp16"):
        raise ValueError(f"{op_name}: dtype 非法 {name!r}（可选 auto/fp32/fp16）")
    return (torch.float32 if n == "fp32" else torch.float16), n


def _check_cuda(op_name):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"{op_name}: 需要 CUDA（IC-Light 只能跑 GPU）")


def _weights_path(variant, override, op_name):
    """权重文件定位（纯 os，先快速失败；缺文件给人话 + 下载指引）。"""
    if variant not in VARIANTS:
        raise ValueError(f"{op_name}: variant 非法 {variant!r}，可选 {sorted(VARIANTS)}")
    if override and str(override).strip():
        path = os.path.expanduser(str(override).strip())
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{op_name}: weights 不存在 {path}")
        return path
    path = os.path.join(ICLIGHT_DIR, VARIANTS[variant]["file"])
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{op_name}: 缺 {VARIANTS[variant]['file']}（{path}）。下载："
            f"huggingface-cli download lllyasviel/ic-light "
            f"{VARIANTS[variant]['file']} --local-dir {ICLIGHT_DIR}"
            f"（权重清单见 nodes/README.md）")
    return path


def _base_dir(op_name):
    path = os.path.expanduser(str(BASE_DIR).strip())
    need = ("model_index.json", "unet", "vae", "text_encoder", "tokenizer")
    missing = [f for f in need if not os.path.exists(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(
            f"{op_name}: SD1.5 基座不完整 {path}（缺 {'、'.join(missing)}）。"
            f"IC-Light 的 delta 权重需要 diffusers 目录格式基座"
            f"（默认 /models/v1-5-pruned-emaonly-fp16）")
    return path


def _register(key, obj, est_bytes, arch, op_name):
    """注册进 model_manager LRU（与整管同一淘汰/pin/unload 语义；flux.py 同款）。"""
    from app.main import models
    with models._lock:
        models._evict_if_needed(est_bytes)
        models._pipes[key] = obj
        models._archs[key] = arch
        models._sizes[key] = est_bytes
        models._last_used[key] = time.time()
    print(f"[{op_name}] registered '{key}' est={est_bytes / 1e9:.2f}GB", flush=True)
    return obj


def _cached(key):
    """取同进程常驻 Bundle（touch LRU）；已被淘汰/未装载返回 None。

    唯一真相源是 `model_manager._pipes`——不要再维护模块级字典（见上方注释）。
    """
    try:
        from app.main import models
    except Exception:
        return None
    with models._lock:
        obj = models._pipes.get(key)
        if obj is not None:
            models._last_used[key] = time.time()
        return obj


def _residents(op_name):
    try:
        from app.main import models
        return "、".join(models.resident()) or "(无)"
    except Exception:
        return "(不可用)"


def _dumps(obj):
    """JSON 负载的信封编码：`@` + JSON（Studio 会把"像 JSON 的字符串"自动反序列化成
    map，注入节点 env 时变成 Go 的 `<map[string]interface {} Value>`，B3 真机复现）。"""
    return "@" + json.dumps(obj, ensure_ascii=False)


def _loads(raw):
    """`@` 信封解码（容忍裸 JSON；非 JSON 原样返回）。"""
    if not isinstance(raw, str):
        return raw
    t = raw[1:] if raw.startswith("@") else raw
    try:
        return json.loads(t)
    except Exception:
        return raw


def _key_parts(model_ref, op_name):
    """`iclight:<variant>` → (键, 变体)；格式错时给可操作报错。"""
    key = str(model_ref or "").strip()
    if not key.startswith("iclight:"):
        raise ValueError(f"{op_name}: model 非法 {model_ref!r}（应为 iclight-load 输出的 "
                         f"'iclight:<variant>'）")
    variant = key.split(":", 1)[1]
    if variant not in VARIANTS:
        raise ValueError(f"{op_name}: model 里的变体非法 {variant!r}，可选 "
                         f"{sorted(VARIANTS)}")
    return key, variant


def _align64(v, op_name, what, lo=64, hi=2048):
    """边长对齐到 64 的倍数（官方 resize 目标都以 64 为步长）。"""
    n = int(v)
    n = max(lo, min(hi, n))
    return n - (n % 64)


def _round64(w, h, op_name):
    """给定时按 64 取整；给 0 时按原尺寸取整到 64 的倍数（不小于 64）。"""
    out = []
    for v, name in ((w, "width"), (h, "height")):
        n = int(v or 0)
        if n <= 0:
            out.append(0)
        else:
            out.append(_align64(n, op_name, name))
    return out[0], out[1]


# ---------------------------------------------------------------------------
# 构建（基座 + conv_in 手术 + delta 权重 + 双条件钩子）
# ---------------------------------------------------------------------------

class Bundle:
    """IC-Light 推理组件束：diffusers 管线 + 调度器 + 元信息。"""

    def __init__(self, variant, dtype_name, unet, vae, text_encoder, tokenizer,
                 t2i, i2i, ddim, dpmpp, info):
        self.variant = variant
        self.dtype_name = dtype_name
        self.unet = unet
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.t2i = t2i
        self.i2i = i2i
        self.ddim = ddim
        self.dpmpp = dpmpp
        self.info = info
        self.est_bytes = info.get("est_mb", 0) * 1024 ** 2


def _build(variant, weights, dtype, dtype_name, op_name):
    """按官方 gradio_demo(_bg).py 的流程装配（返回 Bundle, info）。"""
    import torch
    import safetensors.torch as sf
    from diffusers import (StableDiffusionPipeline, StableDiffusionImg2ImgPipeline,
                           DDIMScheduler, DPMSolverMultistepScheduler)

    V = VARIANTS[variant]
    base = _base_dir(op_name)
    t0 = time.time()
    pipe = StableDiffusionPipeline.from_pretrained(
        base, torch_dtype=dtype, safety_checker=None,
        requires_safety_checker=False, feature_extractor=None)
    pipe.set_progress_bar_config(disable=True)
    unet, vae = pipe.unet, pipe.vae
    te, tok = pipe.text_encoder, pipe.tokenizer
    load_sec = time.time() - t0

    # ---- ① conv_in 扩通道手术（官方逐行等价：zero → 拷前 4 通道 → 复用 bias）----
    with torch.no_grad():
        nc = torch.nn.Conv2d(V["conv_in"], unet.conv_in.out_channels,
                             unet.conv_in.kernel_size, unet.conv_in.stride,
                             unet.conv_in.padding)
        nc.weight.zero_()
        nc.weight[:, :4, :, :].copy_(unet.conv_in.weight)
        nc.bias = unet.conv_in.bias
        unet.conv_in = nc

    # ---- ② IC-Light delta 权重（origin + offset，严格匹配）----
    off = sf.load_file(weights)
    ori = unet.state_dict()
    missing = [k for k in ori if k not in off]
    extra = [k for k in off if k not in ori]
    shape_bad = [(k, tuple(ori[k].shape), tuple(off[k].shape)) for k in ori
                 if k in off and tuple(ori[k].shape) != tuple(off[k].shape)]
    if missing or extra or shape_bad:
        raise ValueError(
            f"{op_name}: IC-Light 权重与基座不匹配（missing={len(missing)} "
            f"extra={len(extra)} shape_bad={len(shape_bad)}）。基座必须是官方 SD1.5"
            f" 结构（{base}）；权重 {weights}。例：missing={missing[:2]} "
            f"shape_bad={shape_bad[:2]}")
    unet.load_state_dict({k: ori[k] + off[k].to(ori[k].dtype) for k in ori},
                         strict=True)
    n_keys = len(ori)
    del off, ori
    torch.cuda.empty_cache()

    # ---- ③ 双条件注入（猴子补丁 unet.forward；官方同款）----
    if not getattr(unet, "_icl_patched", False):
        orig_fwd = unet.forward

        def hooked(sample, timestep, encoder_hidden_states, **kw):
            ck = dict(kw.get("cross_attention_kwargs") or {})
            cc = ck.pop("concat_conds", None)
            kw["cross_attention_kwargs"] = ck
            if cc is not None:
                cc = cc.to(sample)
                # 官方：条件批 1 时按 sample 批扩张（num_images_per_prompt>1）
                cc = torch.cat([cc] * (sample.shape[0] // cc.shape[0]), dim=0)
                sample = torch.cat([sample, cc], dim=1)
            return orig_fwd(sample, timestep, encoder_hidden_states, **kw)

        unet.forward = hooked
        unet._icl_patched = True

    unet.to(device="cuda", dtype=dtype)
    te.to(device="cuda", dtype=dtype)
    vae.to(device="cuda", dtype=dtype)
    for mod in (unet, vae):
        try:
            from diffusers.models.attention_processor import AttnProcessor2_0
            mod.set_attn_processor(AttnProcessor2_0())
        except Exception:
            pass

    ddim = DDIMScheduler(num_train_timesteps=1000, beta_start=0.00085,
                         beta_end=0.012, beta_schedule="scaled_linear",
                         clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    dpmpp = DPMSolverMultistepScheduler(num_train_timesteps=1000,
                                        beta_start=0.00085, beta_end=0.012,
                                        algorithm_type="sde-dpmsolver++",
                                        use_karras_sigmas=True, steps_offset=1)
    kw = dict(vae=vae, text_encoder=te, tokenizer=tok, unet=unet,
              safety_checker=None, requires_safety_checker=False,
              feature_extractor=None, image_encoder=None)
    t2i = StableDiffusionPipeline(scheduler=dpmpp, **kw)
    i2i = StableDiffusionImg2ImgPipeline(scheduler=dpmpp, **kw)
    for p in (t2i, i2i):
        p.set_progress_bar_config(disable=True)

    n_param = (sum(p.numel() for p in unet.parameters())
               + sum(p.numel() for p in vae.parameters())
               + sum(p.numel() for p in te.parameters()))
    est_mb = int(n_param * dtype.itemsize / 1024 ** 2)
    info = {"variant": variant, "dtype": dtype_name, "conv_in": V["conv_in"],
            "keys": n_keys, "load_sec": round(load_sec, 1), "est_mb": est_mb,
            "base": base, "weights": weights}
    print(f"[{op_name}] build {variant} dtype={dtype_name} conv_in={V['conv_in']} "
          f"keys={n_keys} est={est_mb}MB base_load={load_sec:.1f}s", flush=True)
    return Bundle(variant, dtype_name, unet, vae, te, tok, t2i, i2i, ddim, dpmpp,
                  info), info


def _ensure(key, variant, op_name, weights=None, dtype=None):
    """取常驻 Bundle；不在（或精度变了）则现装。"""
    want = _DTYPES.get(key)
    got = _cached(key)
    if got is not None and (dtype is None or want == dtype):
        return got
    if got is not None and want != dtype:      # 精度变了：卸掉重建
        from app.main import models
        models.unload(key)
    _check_cuda(op_name)
    wp = _weights_path(variant, weights, op_name)
    dt, dt_name = _dtype(dtype or want or "auto", op_name)
    t0 = time.time()
    b, info = _build(variant, wp, dt, dt_name, op_name)
    _register(key, b, b.est_bytes, "iclight", op_name)
    _DTYPES[key] = dt_name
    _LOADINFO[key] = info
    print(f"[{op_name}] {variant} dtype={dt_name} 装载 {time.time() - t0:.1f}s "
          f"常驻={_residents(op_name)}", flush=True)
    return b


# ---------------------------------------------------------------------------
# 推理（官方 process() 等价实现）
# ---------------------------------------------------------------------------

def _np2pt(np, imgs):
    import torch
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    return h.movedim(-1, 1)


def _pt2np(np, imgs):
    out = []
    for x in imgs:
        y = x.movedim(0, -1) * 127.5 + 127.5
        out.append(y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8))
    return out


def _resize_center(np, ImageMod, img, tw, th):
    """官方 resize_and_center_crop：先按短边放大到覆盖，再中心裁。"""
    im = ImageMod.fromarray(img)
    w, h = im.size
    s = max(tw / w, th / h)
    rw, rh = int(round(w * s)), int(round(h * s))
    r = im.resize((rw, rh), ImageMod.LANCZOS)
    return np.array(r.crop(((rw - tw) / 2, (rh - th) / 2,
                            (rw + tw) / 2, (rh + th) / 2)))


def _resize_plain(np, ImageMod, img, tw, th):
    return np.array(ImageMod.fromarray(img).resize((tw, th), ImageMod.LANCZOS))


def _grad(np, kind, w, h, lo, hi):
    """光方向梯度（官方：fc 用 0~255，fbc 用 32~224）。"""
    if kind == "left":
        g = np.linspace(hi, lo, w)
        m = np.tile(g, (h, 1))
    elif kind == "right":
        g = np.linspace(lo, hi, w)
        m = np.tile(g, (h, 1))
    elif kind == "top":
        g = np.linspace(hi, lo, h)[:, None]
        m = np.tile(g, (1, w))
    elif kind == "bottom":
        g = np.linspace(lo, hi, h)[:, None]
        m = np.tile(g, (1, w))
    else:
        raise ValueError(kind)
    return np.stack((m,) * 3, axis=-1).astype(np.uint8)


def _encode_prompt_pair(torch, te, tok, pos, neg):
    """官方 encode_prompt_pair：长 prompt 分块 + 正负对齐（避免 77 token 截断）。"""
    import math

    def inner(txt):
        ml = tok.model_max_length
        cl = ml - 2
        ids = tok(txt, truncation=False, add_special_tokens=False)["input_ids"]
        chunks = [[tok.bos_token_id] + ids[i:i + cl] + [tok.eos_token_id]
                  for i in range(0, len(ids), cl)] or [[tok.bos_token_id,
                                                        tok.eos_token_id]]
        chunks = [ck[:ml] if len(ck) >= ml
                  else ck + [tok.eos_token_id] * (ml - len(ck)) for ck in chunks]
        t = torch.tensor(chunks).to(device="cuda", dtype=torch.int64)
        return te(t).last_hidden_state

    c, uc = inner(pos), inner(neg)
    mc = max(len(c), len(uc))
    c = torch.cat([c] * int(math.ceil(mc / len(c))), dim=0)[:mc]
    uc = torch.cat([uc] * int(math.ceil(mc / len(uc))), dim=0)[:mc]
    return (torch.cat([p[None, ...] for p in c], dim=1),
            torch.cat([p[None, ...] for p in uc], dim=1))


def _gray_composite(np, fg, mask):
    """官方 run_rmbg(sigma=0) 的合成公式：127 + (img - 127) * alpha。"""
    a = np.asarray(mask.convert("L").resize(fg.size, Image.NEAREST),
                   dtype=np.float32) / 255.0
    src = np.asarray(fg.convert("RGB"), dtype=np.float32)
    out = 127.0 + (src - 127.0) * a[:, :, None]
    return out.clip(0, 255).astype(np.uint8)


def _infer(b, image, mask, bg, p):
    """官方 process()：低清（t2i/i2i）+ 可选高清（i2i）两阶段。"""
    import numpy as np
    import torch
    from PIL import Image as ImageMod

    with torch.inference_mode():
        op = "iclight.relight"
        fg_pil = image.convert("RGB")
        fg_np = _gray_composite(np, fg_pil, mask) if mask is not None \
            else np.asarray(fg_pil)
        # 目标尺寸：0 → 用前景尺寸，对齐 64 的倍数
        w = p["width"] or fg_np.shape[1]
        h = p["height"] or fg_np.shape[0]
        w = _align64(w, op, "width")
        h = _align64(h, op, "height")
        variant = b.variant
        grads = VARIANTS[variant]["grad"]
        cond = VARIANTS[variant]["cond"]
        light = p["light_dir"]

        # 背景条件：fbc 可给 bg 图；未给则按 light_dir 造梯度；none → 64 灰（官方 GREY）
        bg_np = None
        if cond == 2:
            if bg is not None:
                bg_np = np.asarray(bg.convert("RGB"))
            elif light != "none":
                bg_np = _grad(np, light, w, h, grads[0], grads[1])
            else:
                bg_np = np.zeros((h, w, 3), dtype=np.uint8) + 64

        rng = torch.Generator(device="cuda").manual_seed(int(p["seed"]))
        t0 = time.time()
        vae = b.vae
        fg = _resize_center(np, ImageMod, fg_np, w, h)
        cc = _np2pt(np, [fg]).to(device=vae.device, dtype=vae.dtype)
        cc = vae.encode(cc).latent_dist.mode() * vae.config.scaling_factor
        if cond == 2:
            bgr = _resize_center(np, ImageMod, bg_np, w, h)
            bl = _np2pt(np, [bgr]).to(device=vae.device, dtype=vae.dtype)
            bl = vae.encode(bl).latent_dist.mode() * vae.config.scaling_factor
            cc = torch.cat([c[None, ...] for c in torch.cat([cc, bl], dim=0)], dim=1)
        c, uc = _encode_prompt_pair(torch, b.text_encoder, b.tokenizer,
                                    p["prompt"] + ", " + p["a_prompt"],
                                    p["negative_prompt"])

        common = dict(prompt_embeds=c, negative_prompt_embeds=uc, width=w, height=h,
                      generator=rng, output_type="latent",
                      guidance_scale=p["cfg"],
                      cross_attention_kwargs={"concat_conds": cc})
        if variant == "fc" and light != "none":
            # fc：梯度图作 i2i 初始 latent（官方 lowres_denoise 局部去噪）
            gi = _resize_center(np, ImageMod,
                                _grad(np, light, w, h, grads[0], grads[1]), w, h)
            gl = _np2pt(np, [gi]).to(device=vae.device, dtype=vae.dtype)
            gl = vae.encode(gl).latent_dist.mode() * vae.config.scaling_factor
            lat = b.i2i(image=gl, strength=p["denoise_low"],
                        num_inference_steps=max(1, int(round(p["steps"]
                                                            / p["denoise_low"]))),
                        **common).images.to(vae.dtype) / vae.config.scaling_factor
        else:
            # fc + none：纯 t2i；fbc：官方**恒用 t2i**（背景只作为条件）
            lat = b.t2i(num_inference_steps=p["steps"],
                        **common).images.to(vae.dtype) / vae.config.scaling_factor
        px = _pt2np(np, vae.decode(lat).sample)
        out_size = (w, h)

        if p["highres"]:
            s = float(p["highres_scale"])
            w2 = int(round(w * s / 64.0) * 64)
            h2 = int(round(h * s / 64.0) * 64)
            px = [_resize_plain(np, ImageMod, q, w2, h2) for q in px]
            pl = _np2pt(np, px).to(device=vae.device, dtype=vae.dtype)
            lat = vae.encode(pl).latent_dist.mode() * vae.config.scaling_factor
            lat = lat.to(device=b.unet.device, dtype=b.unet.dtype)
            fg2 = _resize_center(np, ImageMod, fg_np, w2, h2)
            cc2 = _np2pt(np, [fg2]).to(device=vae.device, dtype=vae.dtype)
            cc2 = vae.encode(cc2).latent_dist.mode() * vae.config.scaling_factor
            if cond == 2:
                bg2 = _resize_center(np, ImageMod, bg_np, w2, h2)
                bl2 = _np2pt(np, [bg2]).to(device=vae.device, dtype=vae.dtype)
                bl2 = vae.encode(bl2).latent_dist.mode() * vae.config.scaling_factor
                cc2 = torch.cat([c[None, ...] for c in torch.cat([cc2, bl2], dim=0)],
                                dim=1)
            lat = b.i2i(image=lat, strength=p["denoise_high"],
                        prompt_embeds=c, negative_prompt_embeds=uc, width=w2,
                        height=h2,
                        num_inference_steps=max(1, int(round(p["steps"]
                                                            / p["denoise_high"]))),
                        generator=rng, output_type="latent",
                        guidance_scale=p["cfg"],
                        cross_attention_kwargs={"concat_conds": cc2}
                        ).images.to(vae.dtype) / vae.config.scaling_factor
            px = _pt2np(np, vae.decode(lat).sample)
            out_size = (w2, h2)

        out = ImageMod.fromarray(px[0])
        # 明暗平衡指标（主体区域左/右平均亮度差）——落盘复核判据的直接观测值
        arr = np.asarray(out.convert("RGB"), dtype=np.float32)
        lum = arr @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        if mask is not None:
            mm = np.asarray(mask.convert("L").resize(out.size, ImageMod.NEAREST),
                            dtype=np.float32) > 128
            if int(mm.sum()) < 100:
                mm = np.ones_like(mm, dtype=bool)
        else:
            mm = np.ones_like(lum, dtype=bool)
        half = out.size[0] // 2
        lm = lum[:, :half][mm[:, :half]]
        rm = lum[:, half:][mm[:, half:]]
        lum_l, lum_r = float(lm.mean()), float(rm.mean())
        return out, {"lum_left": round(lum_l, 1), "lum_right": round(lum_r, 1),
                     "light_balance": round(lum_l - lum_r, 1),
                     "size": list(out_size), "sec": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------

def iclight_load(variant="fc", dtype="auto", weights="", reload="false"):
    """装载 IC-Light 变体（SD15 基座 + delta 权重）→ 模型键 `iclight:<variant>`。"""
    op_name = f"iclight.load@{OP_VERSION}"
    variant = str(variant or "fc").strip().lower()
    if variant not in VARIANTS:
        raise ValueError(f"{op_name}: variant 非法 {variant!r}，可选 {sorted(VARIANTS)}")
    key = f"iclight:{variant}"
    want_reload = str(reload).strip().lower() in ("1", "true", "yes", "on")
    if want_reload:
        from app.main import models
        models.unload(key)
        _DTYPES.pop(key, None)
    wp = _weights_path(variant, weights, op_name)     # 先快速失败
    _base_dir(op_name)
    dt, dt_name = _dtype(dtype, op_name)
    was = _cached(key)
    b = _ensure(key, variant, op_name, weights=weights, dtype=dt_name)
    info = dict(_LOADINFO.get(key) or b.info)
    info["cache_hit"] = was is not None
    print(f"[{op_name}] {variant} dtype={dt_name} est={b.est_bytes / 1024 ** 2:.0f}MB "
          f"命中={was is not None} 常驻={_residents(op_name)}", flush=True)
    return {"model": key, "params_mb": int(b.est_bytes // 1024 ** 2),
            "vram_mb": int(b.est_bytes // 1024 ** 2), "info": _dumps(info)}


def iclight_relight(image, model, mask=None, bg=None, light_dir="none",
                    prompt="beautiful woman, detailed face, soft studio lighting",
                    negative_prompt="lowres, bad anatomy, bad hands, cropped, "
                                    "worst quality",
                    a_prompt="best quality", width=0, height=0, steps=25, cfg=2.0,
                    seed=12345, denoise_low=0.9, denoise_high=0.5,
                    highres_scale=1.5, highres="true", scheduler="dpmpp"):
    """重打光：前景（可带 mask 自动合成 127 灰底）+ 光方向/背景 → 重打光图。"""
    op_name = f"iclight.relight@{OP_VERSION}"
    key, variant = _key_parts(model, op_name)
    light = str(light_dir or "none").strip().lower()
    if light not in LIGHT_DIRS:
        raise ValueError(f"{op_name}: light_dir 非法 {light_dir!r}，可选 "
                         f"{list(LIGHT_DIRS)}")
    sched = str(scheduler or "dpmpp").strip().lower()
    if sched not in ("dpmpp", "ddim"):
        raise ValueError(f"{op_name}: scheduler 非法 {scheduler!r}（可选 dpmpp/ddim）")
    if variant == "fc" and bg is not None:
        print(f"[{op_name}] 提示：fc 变体没有背景条件通道，bg 已忽略"
              f"（要按背景打光请用 fbc）", flush=True)
    n_steps = int(steps)
    if not (1 <= n_steps <= 100):
        raise ValueError(f"{op_name}: steps 非法 {steps!r}（需 1~100）")
    w, h = _round64(width, height, op_name)
    p = {"light_dir": light, "prompt": str(prompt or "").strip(),
         "negative_prompt": str(negative_prompt or "").strip(),
         "a_prompt": str(a_prompt or "").strip(), "cfg": float(cfg),
         "seed": int(seed), "steps": n_steps,
         "denoise_low": float(denoise_low), "denoise_high": float(denoise_high),
         "highres": str(highres).strip().lower() in ("1", "true", "yes", "on"),
         "highres_scale": float(highres_scale), "width": w, "height": h}
    if not (0.1 <= p["denoise_low"] <= 1.0) or not (0.1 <= p["denoise_high"] <= 1.0):
        raise ValueError(f"{op_name}: denoise 非法（需 0.1~1.0）")
    if not (1.0 <= p["highres_scale"] <= 3.0):
        raise ValueError(f"{op_name}: highres_scale 非法 {highres_scale!r}（需 1~3）")
    if not (1.0 <= p["cfg"] <= 32.0):
        raise ValueError(f"{op_name}: cfg 非法 {cfg!r}（需 1~32）")
    b = _ensure(key, variant, op_name)
    b.t2i.scheduler = b.dpmpp if sched == "dpmpp" else b.ddim
    b.i2i.scheduler = b.t2i.scheduler
    out, m = _infer(b, image, mask, bg, p)
    ow, oh = out.size
    info = {"variant": variant, "dtype": b.dtype_name, "light_dir": light,
            "scheduler": sched, "steps": n_steps, "cfg": p["cfg"], "seed": p["seed"],
            "highres": p["highres"], "highres_scale": p["highres_scale"],
            "denoise_low": p["denoise_low"], "denoise_high": p["denoise_high"],
            "fg_mask": mask is not None, "bg_given": bg is not None,
            "out_size": [ow, oh], "lum_left": m["lum_left"],
            "lum_right": m["lum_right"], "light_balance": m["light_balance"],
            "sec": m["sec"]}
    print(f"[{op_name}] {variant} light={light} {image.size[0]}x{image.size[1]} → "
          f"{ow}x{oh} steps={n_steps} seed={p['seed']} 明暗平衡="
          f"{m['light_balance']:+.1f}（左 {m['lum_left']}/右 {m['lum_right']}）"
          f" {m['sec']}s", flush=True)
    return {"image": out, "width": int(ow), "height": int(oh),
            "light_balance": float(m["light_balance"]),
            "lum_left": float(m["lum_left"]), "lum_right": float(m["lum_right"]),
            "info": _dumps(info)}


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "iclight.load",
        inputs={"variant": "STRING", "dtype": "STRING", "weights": "STRING",
                "reload": "STRING"},
        outputs={"model": "STRING", "params_mb": "INT", "vram_mb": "INT",
                 "info": "STRING"},
        description="装载 IC-Light 重打光模型（variant=fc|fbc；权重在 "
                    "ICLIGHT_MODELS_DIR/<variant>.safetensors，基座为 diffusers 目录格式 "
                    "SD1.5，默认 /models/v1-5-pruned-emaonly-fp16）→ 模型键 "
                    "'iclight:<variant>'，注册进 model_manager LRU。dtype=auto 在算力<7.0 "
                    "取 fp32（Pascal fp16 吞吐 1/64）；reload=true 强制重装"
    )(iclight_load)
    registry.register(
        "iclight.relight",
        inputs={"image": "IMAGE", "model": "STRING", "mask": "IMAGE", "bg": "IMAGE",
                "light_dir": "STRING", "prompt": "STRING",
                "negative_prompt": "STRING", "a_prompt": "STRING", "width": "INT",
                "height": "INT", "steps": "INT", "cfg": "FLOAT", "seed": "INT",
                "denoise_low": "FLOAT", "denoise_high": "FLOAT",
                "highres_scale": "FLOAT", "highres": "STRING",
                "scheduler": "STRING"},
        outputs={"image": "IMAGE", "width": "INT", "height": "INT",
                 "light_balance": "FLOAT", "lum_left": "FLOAT",
                 "lum_right": "FLOAT", "info": "STRING"},
        description="IC-Light 重打光（G9）：前景（image；给 mask 则按官方公式合成 127 "
                    "灰底）按 light_dir=none|left|right|top|bottom 或 bg 图重打光。"
                    "fc 变体用 light_dir 造初始图；fbc 变体把 bg 图作为背景条件"
                    "（未给 bg 时 light_dir 造 32~224 梯度、none 用 64 灰）。"
                    "light_balance 是输出主体左/右平均亮度差（落盘复核判据）；"
                    "同种子逐字节可复现"
    )(iclight_relight)
