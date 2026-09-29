"""FLUX 细分节点链服务端算子插件（ComfyUI 风格组件级编排，随节点包自注册）。

与旧聚合插件（flux.load/flux.encode/flux.sample 单管 FluxPipeline）的区别：
组件拆成独立对象在节点间传递——UNET（transformer GGUF）/ CLIP（CLIP-L+T5 束）
/ VAE / COND / LATENT，采样器吃 latent 吐 latent，VAE 编解码独立成节点，
可编排 hires-fix（vae_encode → sample denoise<1）、latent 级接力等 ComfyUI
同款图。LATENT 束：{samples, packed, width, height, batch}——txt2img 路径
samples=None（采样器按 seed 生噪声）；vae_encode 路径为未 pack 的归一化张量。

内存相位（12GB 内存盒核心约束，继承聚合版设计）：
- 组件各自注册进 model_manager LRU（flux-unet:/flux-clip:/flux-vae: 键），
  淘汰/pin/unload 语义与整管一致（对象身份比较，见 model_manager.keys_of）
- flux.encode：先把所有 FluxTransformer2DModel 搬回 CPU（6.8GB 让位）→
  T5 装载上卡（bf16 4.9GB，GGUF→npy 分片缓存，首建 ~4min 此后 ~40s）→
  编码 → release_t5=1（默认）释放 / 0 搬回 CPU
- flux.sample：transformer 上卡（~6.8GB），采样 output_type="latent" 不解码
  （VAE 可不上卡）；完事留卡供连续采样
- flux.vae_decode/encode、flux.detail_refine：VAE 小（~0.3GB）随用随上卡

img2img（latent + denoise<1）实现要点（对拍 diffusers 0.35.2 源码）：
- FluxPipeline.__call__ 支持 sigmas 自定义调度 + latents 直通（须已 pack）；
  shift/time_shift 均为逐点函数 ⇒ shifted(slice) == slice(shifted)，传
  pre-shift linspace 尾部切片即可精确复刻 FluxImg2ImgPipeline 的 timesteps 尾段
- 加噪手动做 flow-match 前向：x_t = (1-σ)x + σ·noise，σ = 完整调度 sigmas[t_start]
  （diffusers img2img 对传入 latents 不加噪，必须自己加）
- denoise>=1 且给了 latent：等价重生成（latents=None 纯噪声起步）

算子：
- flux.unet_load(transformer) → MODEL（FluxTransformer2DModel GGUF Q4）
- flux.dual_clip_load(t5) → CLIP（FluxClipBundle：CLIP-L+双 tokenizer+T5 惰性信息）
- flux.vae_load(name) → VAE（AutoencoderKL）
- flux.encode(clip, text, max_seq, release_t5) → COND{prompt_embeds, pooled}
- flux.empty_latent(width, height, batch_size) → LATENT（samples=None）
- flux.sample(model, cond, latent, seed, steps, guidance, denoise,
  preview_every) → LATENT（packed）
- flux.vae_decode(vae, latent) → IMAGE
- flux.vae_encode(vae, image) → LATENT（未 pack，归一化）
- flux.detail_refine(model, vae, cond, image, ...) → IMAGE（ADetailer 式，
  img2img 用 FluxImg2ImgPipeline 按组件装配，PIL crop 路径天然正确）
"""
import glob
import json
import os
import random
import time

import torch

from app import execution, ops, preview

FLUX_DIR = os.environ.get("FLUX_DIR", "/models/flux")
DETECTOR_DIR = os.environ.get("DETECTOR_DIR", "/models/detectors")

# ---------------------------------------------------------------------------
# torch 2.3.1 兼容补丁（幂等，模块导入即安装；与旧聚合插件相同三处）
# ---------------------------------------------------------------------------


def _install_torch_compat():
    if not hasattr(torch.nn, "RMSNorm"):
        class _RMSNorm(torch.nn.Module):
            def __init__(self, dim, eps=1e-6, elementwise_affine=True):
                super().__init__()
                self.eps = eps
                self.weight = (torch.nn.Parameter(torch.ones(dim))
                               if elementwise_affine else None)

            def forward(self, x):
                dt = x.dtype
                v = x.float()
                v = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + self.eps)
                v = v.to(dt)
                if self.weight is not None:
                    v = v * self.weight.to(dt)
                return v

        torch.nn.RMSNorm = _RMSNorm
        print("[flux] torch.nn.RMSNorm patched (torch<2.4)", flush=True)

    import torch.nn.functional as F
    if not getattr(F.scaled_dot_product_attention, "_flux_gqa_shim", False):
        _orig = F.scaled_dot_product_attention

        def _sdpa_shim(*a, **kw):
            gqa = kw.pop("enable_gqa", None)
            if gqa:
                raise RuntimeError(
                    "enable_gqa=True 需要 torch>=2.5（本 shim 仅剥离 False）")
            return _orig(*a, **kw)

        _sdpa_shim._flux_gqa_shim = True
        F.scaled_dot_product_attention = _sdpa_shim
        print("[flux] SDPA enable_gqa shim applied (torch<2.5)", flush=True)


_install_torch_compat()

# ---------------------------------------------------------------------------
# T5-XXL GGUF 手工装载（transformers 4.44 不支持 T5 架构 GGUF；与旧版同实现）
# ---------------------------------------------------------------------------

_T5_TENSOR_MAP = {
    "attn_q.weight": "layer.0.SelfAttention.q.weight",
    "attn_k.weight": "layer.0.SelfAttention.k.weight",
    "attn_v.weight": "layer.0.SelfAttention.v.weight",
    "attn_o.weight": "layer.0.SelfAttention.o.weight",
    "attn_rel_b.weight": "layer.0.SelfAttention.relative_attention_bias.weight",
    "attn_norm.weight": "layer.0.layer_norm.weight",
    "ffn_gate.weight": "layer.1.DenseReluDense.wi_0.weight",
    "ffn_up.weight": "layer.1.DenseReluDense.wi_1.weight",
    "ffn_down.weight": "layer.1.DenseReluDense.wo.weight",
    "ffn_norm.weight": "layer.1.layer_norm.weight",
}

_T5_CACHE_MARK = "COMPLETE"


def _t5_hf_name(gguf_name):
    if gguf_name == "token_embd.weight":
        return "shared.weight"
    if gguf_name == "enc.output_norm.weight":
        return "encoder.final_layer_norm.weight"
    if gguf_name.startswith("enc.blk."):
        rest = gguf_name[len("enc.blk."):]
        i, name = rest.split(".", 1)
        tgt = _T5_TENSOR_MAP.get(name)
        if tgt:
            return f"encoder.block.{int(i)}.{tgt}"
    return None


def _t5_cache_fname(hf_name):
    return hf_name.replace(".", "__") + ".npy"


def _t5_state_dict_from_gguf(gguf_path, dtype, device="cpu", cache_dir=None):
    """GGUF → HF T5 state_dict。gguf shape 是 [in,out]，数据行主序按 [out,in]
    存 → reshape(shape[::-1])。cache_dir 逐张量 npy 落盘（safetensors 全量
    副本会在 12GB 内存盒 OOM）。"""
    import numpy as np
    from gguf import GGUFReader
    from gguf.quants import dequantize

    r = GGUFReader(gguf_path)
    sd = {}
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    for t in r.tensors:
        hf = _t5_hf_name(t.name)
        if hf is None:
            continue
        arr = dequantize(t.data, t.tensor_type).reshape(t.shape[::-1])
        if cache_dir:
            np.save(os.path.join(cache_dir, _t5_cache_fname(hf)),
                    arr.astype(np.float32))
        w = torch.from_numpy(np.ascontiguousarray(arr.copy())).to(dtype)
        if device != "cpu":
            w = w.to(device)
        sd[hf] = w
        del arr, w
    if cache_dir:
        open(os.path.join(cache_dir, _T5_CACHE_MARK), "w").write("ok")
    return sd


def _t5_state_dict_from_cache(cache_dir, dtype, device):
    import numpy as np
    sd = {}
    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith(".npy"):
            continue
        hf = fn[:-4].replace("__", ".")
        arr = np.load(os.path.join(cache_dir, fn))
        w = torch.from_numpy(arr).to(dtype)
        if device != "cpu":
            w = w.to(device)
        sd[hf] = w
        del arr, w
    return sd


def load_t5_gguf(gguf_path, config_path, dtype=torch.bfloat16, device="cpu"):
    """GGUF T5 → T5EncoderModel（bf16 默认；fp16 高激活崩坏勿改）。
    meta 壳 + assign=True；逐张量 npy 缓存。device="cuda" 逐张量直上卡。"""
    from transformers import T5Config, T5EncoderModel

    cache_dir = gguf_path + f".{str(dtype).split('.')[-1]}.cache"
    cfg = T5Config(**json.load(open(config_path)))
    if os.path.isfile(os.path.join(cache_dir, _T5_CACHE_MARK)):
        t0 = time.time()
        sd = _t5_state_dict_from_cache(cache_dir, dtype, device)
        print(f"[flux] t5 缓存命中（{time.time()-t0:.1f}s）", flush=True)
    else:
        t0 = time.time()
        write_cache = (device == "cpu")
        sd = _t5_state_dict_from_gguf(
            gguf_path, dtype, device=device,
            cache_dir=cache_dir if write_cache else None)
        print(f"[flux] t5 gguf->sd {time.time()-t0:.1f}s tensors={len(sd)}"
              f"{'（缓存已写）' if write_cache else ''}", flush=True)
    sd.setdefault("encoder.embed_tokens.weight", sd["shared.weight"])
    with torch.device("meta"):
        model = T5EncoderModel(cfg)
    missing, unexpected = model.load_state_dict(sd, assign=True, strict=False)
    missing = [m for m in missing if "relative_attention_bias" not in m]
    if missing or unexpected:
        print(f"[flux] WARN t5 load missing={missing[:5]} "
              f"unexpected={unexpected[:5]}", flush=True)
    del sd
    model.tie_weights()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 组件注册（model_manager LRU；与整管同一淘汰/pin/unload 语义）
# ---------------------------------------------------------------------------


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
    print(f"[flux] registered '{key}' est={est_bytes/1e9:.1f}GB", flush=True)
    return obj


def _require_file(path, what):
    if not os.path.isfile(path):
        raise ValueError(f"flux: 缺 {what}（{path}）")
    return path


class FluxClipBundle:
    """DualCLIP 组件束：CLIP-L 模块 + 双 tokenizer + T5 惰性装载信息。
    T5 权重在编码相位才装载（内存错峰），release_t5=1 编码后释放。"""

    def __init__(self, clip_l, tokenizer, tokenizer_2, t5_gguf, t5_cfg):
        self.clip_l = clip_l
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.t5_gguf = t5_gguf
        self.t5_cfg = t5_cfg
        self.t5 = None


def _move_transformers(device):
    """把所有已注册 FluxTransformer2DModel 搬到指定设备（编码相位腾 VRAM）。"""
    n = 0
    for p in list(_models()._pipes.values()):
        if type(p).__name__ == "FluxTransformer2DModel":
            p.to(device)
            n += 1
    if n:
        torch.cuda.empty_cache()
        print(f"[flux] {n} transformer -> {device}", flush=True)


def _fresh_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(FLUX_DIR, "scheduler"))


def _assemble_txt2img(unet):
    """轻量 FluxPipeline：仅 scheduler+transformer（采样 output_type=latent，
    不需要 VAE/文本组件；vae=None 时 vae_scale_factor 守卫默认 8，正确）。"""
    from diffusers import FluxPipeline
    pipe = FluxPipeline(scheduler=_fresh_scheduler(), vae=None,
                        text_encoder=None, tokenizer=None, text_encoder_2=None,
                        tokenizer_2=None, transformer=unet)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def _flux_latent_preview_pil(pipe, latents, height, width, max_size=256):
    """采样中间 packed latent → PIL 预览帧（通道分组均值投影，粗糙但结构可辨）。"""
    from PIL import Image
    with torch.no_grad():
        h2 = 2 * (int(height) // (pipe.vae_scale_factor * 2))
        w2 = 2 * (int(width) // (pipe.vae_scale_factor * 2))
        lat = pipe._unpack_latents(latents, h2, w2,
                                   pipe.vae_scale_factor)[0].float()
        rgb = torch.stack([lat[0:6].mean(0), lat[6:11].mean(0),
                           lat[11:16].mean(0)])
        lo, hi = rgb.min(), rgb.max()
        rgb = (rgb - lo) / (hi - lo + 1e-6)
        arr = (rgb * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr)
    scale = max_size / max(img.size)
    if scale > 1:
        img = img.resize((round(img.width * scale),
                          round(img.height * scale)), Image.BILINEAR)
    return img


_DETECTOR_CACHE = {}


def _detector(name):
    """YOLO 检测器缓存（与 detail.refine 同一约定，ultralytics 惰性导入）。"""
    if name not in _DETECTOR_CACHE:
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
                "flux.detail_refine 需要 ultralytics") from e
        _DETECTOR_CACHE[name] = YOLO(path)
    return _DETECTOR_CACHE[name]


def _small(img, max_size=512):
    from PIL import Image
    scale = max_size / max(img.size)
    if scale < 1:
        img = img.resize((round(img.width * scale), round(img.height * scale)),
                         Image.Resampling.BILINEAR)
    return img


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------


def register(registry):
    @registry.register(
        "flux.unet_load",
        inputs={"transformer": "STRING"},
        outputs={"model": "MODEL"},
        description="FLUX UNET Loader（对标 ComfyUI UNETLoader）：GGUF Q4_K_S "
                    "transformer 单独装载（fp16 compute），输出 MODEL 组件引用，"
                    "进 model_manager LRU。权重布局 /models/flux，见插件 docstring")
    def _op_flux_unet_load(transformer="flux1-schnell-Q4_K_S.gguf"):
        from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
        key = f"flux-unet:{transformer}"
        models = _models()
        if key in models._pipes:
            return {"model": models.get(key)}
        tr_path = _require_file(os.path.join(FLUX_DIR, transformer),
                                "transformer GGUF")
        tr_cfg = _require_file(
            os.path.join(FLUX_DIR, "transformer", "config.json"),
            "transformer config")
        t0 = time.time()
        quant = GGUFQuantizationConfig(compute_dtype=torch.float16)
        tr = FluxTransformer2DModel.from_single_file(
            tr_path, config=os.path.dirname(tr_cfg),
            quantization_config=quant, torch_dtype=torch.float16)
        print(f"[flux] transformer {transformer} {time.time()-t0:.1f}s",
              flush=True)
        est = os.path.getsize(tr_path)
        return {"model": _register(key, tr, est, "FluxTransformer2DModel(GGUF)")}

    @registry.register(
        "flux.dual_clip_load",
        inputs={"t5": "STRING"},
        outputs={"clip": "CLIP"},
        description="FLUX DualCLIP Loader（对标 ComfyUI DualCLIPLoader）：CLIP-L "
                    "(bf16) + 双 tokenizer 立即装载，T5-XXL(GGUF) 编码相位才装载"
                    "上卡（内存错峰）。输出 CLIP 组件束引用")
    def _op_flux_dual_clip_load(t5="t5xxl-Q4_K_S.gguf"):
        from transformers import CLIPTextModel, CLIPTokenizer, T5TokenizerFast
        key = f"flux-clip:t5={t5}"
        models = _models()
        if key in models._pipes:
            return {"clip": models.get(key)}
        t5_path = _require_file(os.path.join(FLUX_DIR, t5), "T5 GGUF")
        t5_cfg = _require_file(
            os.path.join(FLUX_DIR, "text_encoder_2", "config.json"), "T5 config")
        t0 = time.time()
        bundle = FluxClipBundle(
            clip_l=CLIPTextModel.from_pretrained(
                os.path.join(FLUX_DIR, "text_encoder"),
                torch_dtype=torch.bfloat16),
            tokenizer=CLIPTokenizer.from_pretrained(
                os.path.join(FLUX_DIR, "tokenizer")),
            tokenizer_2=T5TokenizerFast.from_pretrained(
                os.path.join(FLUX_DIR, "tokenizer_2")),
            t5_gguf=t5_path, t5_cfg=t5_cfg)
        print(f"[flux] dual-clip (clip-l + tokenizers, t5 lazy) "
              f"{time.time()-t0:.1f}s", flush=True)
        return {"clip": _register(key, bundle, int(0.5e9), "FluxClipBundle")}

    @registry.register(
        "flux.vae_load",
        inputs={"name": "STRING"},
        outputs={"vae": "VAE"},
        description="FLUX VAE Loader（对标 ComfyUI Load VAE）：装载 FLUX 专用 "
                    "AutoencoderKL（FLUX_DIR/vae，fp16），输出 VAE 组件引用")
    def _op_flux_vae_load(name="vae"):
        from diffusers import AutoencoderKL
        key = f"flux-vae:{name}"
        models = _models()
        if key in models._pipes:
            return {"vae": models.get(key)}
        path = os.path.join(FLUX_DIR, name)
        if not os.path.isdir(path):
            raise ValueError(f"flux: 缺 VAE 目录（{path}）")
        t0 = time.time()
        vae = AutoencoderKL.from_pretrained(path, torch_dtype=torch.float16)
        print(f"[flux] vae {name} {time.time()-t0:.1f}s", flush=True)
        return {"vae": _register(key, vae, int(0.4e9), "AutoencoderKL(FLUX)")}

    @registry.register(
        "flux.encode",
        inputs={"clip": "CLIP", "text": "STRING", "max_seq": "INT",
                "release_t5": "INT"},
        outputs={"cond": "COND", "info": "STRING"},
        description="FLUX CLIP Text Encode（对标 ComfyUI CLIP Text Encode (Flux)）："
                    "CLIP-L pooled(768) + T5-XXL 序列 → COND。编码相位：transformer "
                    "搬回 CPU → T5 上卡 → 编码 → release_t5=1（默认）释放 T5。"
                    "FLUX 免负向提示词（schnell 蒸馏，guidance=0）")
    def _op_flux_encode(clip, text, max_seq=256, release_t5=1):
        from diffusers import FluxPipeline
        # 鸭子校验（插件热更后类身份失效，isinstance 跨版本误杀旧对象）
        if not (hasattr(clip, "clip_l") and hasattr(clip, "t5_gguf")):
            raise ValueError("flux.encode: clip 需为 flux.dual_clip_load 的输出")
        if not torch.cuda.is_available():
            raise ValueError("flux.encode 需要 CUDA（T5 bf16 编码上卡）")
        t0 = time.time()
        # 相位切换：transformer 回 CPU 腾 VRAM，T5/CLIP-L 上卡
        _move_transformers("cpu")
        if clip.t5 is None:
            clip.t5 = load_t5_gguf(clip.t5_gguf, clip.t5_cfg, device="cuda")
        else:
            clip.t5.to("cuda")
        clip.clip_l.to("cuda")
        pipe = FluxPipeline(scheduler=None, vae=None,
                            text_encoder=clip.clip_l, tokenizer=clip.tokenizer,
                            text_encoder_2=clip.t5,
                            tokenizer_2=clip.tokenizer_2, transformer=None)
        with torch.no_grad():
            pe, pooled, _ = pipe.encode_prompt(
                prompt=str(text), prompt_2=None, device=torch.device("cuda"),
                max_sequence_length=int(max_seq))
        del pipe
        clip.clip_l.to("cpu")
        if int(release_t5):
            clip.t5 = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print("[flux] t5 released (release_t5=1)", flush=True)
        else:
            clip.t5.to("cpu")
        info = f"{time.time()-t0:.1f}s pe={tuple(pe.shape)}"
        print(f"[flux.encode] {info}", flush=True)
        return {"cond": {"prompt_embeds": pe.float().cpu(),
                         "pooled": pooled.float().cpu()},
                "info": info}

    @registry.register(
        "flux.empty_latent",
        inputs={"width": "INT", "height": "INT", "batch_size": "INT"},
        outputs={"latent": "LATENT"},
        description="FLUX Empty Latent（对标 ComfyUI Empty Latent Image）：只携带"
                    "尺寸/批次信息（samples=None），噪声由采样器按 seed 生成。"
                    "宽高须 16 倍数")
    def _op_flux_empty_latent(width=768, height=768, batch_size=1):
        return {"latent": {"samples": None, "packed": True,
                           "width": int(width), "height": int(height),
                           "batch": int(batch_size)}}

    @registry.register(
        "flux.sample",
        inputs={"model": "MODEL", "cond": "COND", "latent": "LATENT",
                "seed": "INT", "steps": "INT", "guidance": "FLOAT",
                "denoise": "FLOAT", "preview_every": "INT"},
        outputs={"latent": "LATENT", "seed": "INT"},
        description="FLUX Sampler（对标 ComfyUI SamplerCustomAdvanced，schnell "
                    "蒸馏 guidance=0）：吃 latent 吐 latent（output_type=latent，"
                    "VAE 解码独立成节点）。latent.samples=None → txt2img 全调度；"
                    "samples 有值（vae_encode 产物）且 denoise<1 → img2img 尾段"
                    "调度（手动 flow-match 加噪，对拍 diffusers 语义）；denoise>=1 "
                    "给 latent 等价重生成。预览：逐步 latent 投影帧 + job 进度")
    def _op_flux_sample(model, cond, latent, seed=-1, steps=4, guidance=0.0,
                        denoise=1.0, preview_every=1):
        import numpy as np
        from diffusers import FluxPipeline
        from diffusers.pipelines.flux.pipeline_flux import calculate_shift

        if type(model).__name__ != "FluxTransformer2DModel":
            raise ValueError("flux.sample: model 需为 flux.unet_load 的输出 "
                             f"（当前 {type(model).__name__}）")
        if not isinstance(cond, dict) or "prompt_embeds" not in cond:
            raise ValueError("flux.sample: cond 需为 flux.encode 的输出")
        if not isinstance(latent, dict) or "width" not in latent:
            raise ValueError("flux.sample: latent 需为 flux.empty_latent / "
                             "flux.vae_encode 的输出束")
        if not torch.cuda.is_available():
            raise ValueError("flux.sample 需要 CUDA")

        seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**32 - 1)
        steps = int(steps)
        denoise = float(denoise)
        W, H = int(latent["width"]), int(latent["height"])
        batch = int(latent.get("batch", 1))
        samples = latent.get("samples")
        job = execution.current()
        rec = (preview.recorder_for(job.id, int(preview_every))
               if job and int(preview_every) > 0 else None)

        # 相位切换：transformer 上卡（采样不需要文本组件/VAE）
        model.to("cuda")
        torch.cuda.empty_cache()
        pipe = _assemble_txt2img(model)
        pe = cond["prompt_embeds"].to("cuda", torch.float16)
        pooled = cond["pooled"].to("cuda", torch.float16)
        gen = torch.Generator("cpu").manual_seed(seed)

        def _cb(p, i, t, kw):
            execution.check_cancelled()
            total = p.num_timesteps
            if job:
                job.set_progress(i + 1, total)
            if rec and rec.want(i, total):
                latents = kw.get("latents")
                try:
                    rec.push_pil(
                        _flux_latent_preview_pil(p, latents, H, W),
                        (i + 1) / total)
                except Exception:
                    rec.push_pil(preview.progress_card(i + 1, total),
                                 (i + 1) / total)
            return kw

        t0 = time.time()
        try:
            if samples is None:
                # ---- txt2img：全调度，latents=None 由管道按 seed 生噪声 ----
                out = pipe(
                    prompt_embeds=pe, pooled_prompt_embeds=pooled,
                    num_inference_steps=steps, guidance_scale=float(guidance),
                    width=W, height=H, num_images_per_prompt=batch,
                    generator=gen, output_type="latent",
                    callback_on_step_end=_cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images  # output_type=latent 时 images 即 latent 张量本体
                # （不是 list！.images[0] 会吃掉 batch 维）
            elif denoise >= 1.0:
                # ---- latent + denoise=1：等价重生成（纯噪声起步）----
                print("[flux.sample] latent 输入但 denoise>=1：按重生成处理",
                      flush=True)
                out = pipe(
                    prompt_embeds=pe, pooled_prompt_embeds=pooled,
                    num_inference_steps=steps, guidance_scale=float(guidance),
                    width=W, height=H, num_images_per_prompt=batch,
                    generator=gen, output_type="latent",
                    callback_on_step_end=_cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images  # output_type=latent 时 images 即 latent 张量本体
                # （不是 list！.images[0] 会吃掉 batch 维）
            else:
                # ---- img2img：尾段调度 + 手动 flow-match 加噪 ----
                seq = (H // 8 // 2) * (W // 8 // 2)
                cfg = pipe.scheduler.config
                mu = calculate_shift(
                    seq, cfg.get("base_image_seq_len", 256),
                    cfg.get("max_image_seq_len", 4096),
                    cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
                pipe.scheduler.set_timesteps(steps, device="cpu", mu=mu)
                t_start = int(max(steps - min(steps * denoise, steps), 0))
                if steps - t_start < 1:
                    raise ValueError(
                        f"denoise={denoise} × steps={steps} 后有效步数 < 1，"
                        "请提高 denoise 或 steps")
                sigma0 = float(pipe.scheduler.sigmas[t_start])
                # 重新装配（scheduler 已被 set_timesteps 污染调度状态，换新的；
                # sigmas 由 __call__ 内部 retrieve_timesteps 重新设置）
                pipe = _assemble_txt2img(model)
                s = samples.float()
                noise = torch.randn(s.shape, generator=gen, dtype=torch.float32)
                noised = (1.0 - sigma0) * s + sigma0 * noise  # flow-match 前向
                packed = FluxPipeline._pack_latents(
                    noised.to(torch.float16), s.shape[0], 16, H // 8, W // 8)
                sigmas_pre = np.linspace(1.0, 1.0 / steps, steps)[t_start:]
                out = pipe(
                    prompt_embeds=pe, pooled_prompt_embeds=pooled,
                    num_inference_steps=None, sigmas=sigmas_pre,
                    guidance_scale=float(guidance),
                    width=W, height=H, generator=gen, latents=packed,
                    output_type="latent",
                    callback_on_step_end=_cb,
                    callback_on_step_end_tensor_inputs=["latents"],
                ).images  # latent 张量本体（.images[0] 会吃掉 batch 维）
        except Exception:
            import traceback
            print("[flux.sample] ERROR:", traceback.format_exc(), flush=True)
            raise
        print(f"[flux.sample] done in {time.time()-t0:.1f}s {W}x{H} "
              f"steps={steps} denoise={denoise} seed={seed}", flush=True)
        return {"latent": {"samples": out.cpu(), "packed": True,
                           "width": W, "height": H,
                           "batch": out.shape[0]},
                "seed": seed}

    @registry.register(
        "flux.vae_decode",
        inputs={"vae": "VAE", "latent": "LATENT"},
        outputs={"image": "IMAGE"},
        description="FLUX VAE Decode（对标 ComfyUI VAE Decode）：packed/unpacked "
                    "latent → 图像（unpack + 反归一化 (z/scaling)+shift → decode）")
    def _op_flux_vae_decode(vae, latent):
        from diffusers import FluxPipeline
        from diffusers.image_processor import VaeImageProcessor
        if not isinstance(latent, dict) or latent.get("samples") is None:
            raise ValueError("flux.vae_decode: latent 需为 flux.sample/"
                             "flux.vae_encode 的输出（含 samples）")
        vae.to("cuda")
        s = latent["samples"]
        if latent.get("packed", True):
            z = FluxPipeline._unpack_latents(
                s, int(latent["height"]), int(latent["width"]), 8)
        else:
            z = s
        with torch.no_grad():
            z = (z.to("cuda", vae.dtype) / vae.config.scaling_factor) \
                + vae.config.shift_factor
            img = vae.decode(z, return_dict=False)[0]
        proc = VaeImageProcessor(vae_scale_factor=16)
        pil = proc.postprocess(img, output_type="pil")
        print(f"[flux.vae_decode] {latent['width']}x{latent['height']} -> IMAGE",
              flush=True)
        return {"image": pil[0] if len(pil) == 1 else pil}

    @registry.register(
        "flux.vae_encode",
        inputs={"vae": "VAE", "image": "IMAGE"},
        outputs={"latent": "LATENT"},
        description="FLUX VAE Encode（对标 ComfyUI VAE Encode）：图像 → 未 pack "
                    "归一化 latent（(z-shift)*scaling），供 flux.sample 以 "
                    "denoise<1 做 img2img / hires-fix")
    def _op_flux_vae_encode(vae, image):
        from diffusers.image_processor import VaeImageProcessor
        vae.to("cuda")
        proc = VaeImageProcessor(vae_scale_factor=16)
        images = image if isinstance(image, list) else [image]
        tensors = [proc.preprocess(img) for img in images]
        img_t = torch.cat(tensors).to(device="cuda", dtype=vae.dtype)
        with torch.no_grad():
            z = vae.encode(img_t).latent_dist.sample()
        z = (z - vae.config.shift_factor) * vae.config.scaling_factor
        W, H = images[0].size
        print(f"[flux.vae_encode] {W}x{H} -> LATENT {tuple(z.shape)}", flush=True)
        return {"latent": {"samples": z.float().cpu(), "packed": False,
                           "width": W, "height": H, "batch": z.shape[0]}}

    @registry.register(
        "flux.detail_refine",
        inputs={"model": "MODEL", "vae": "VAE", "cond": "COND", "image": "IMAGE",
                "detector": "STRING", "conf": "FLOAT", "padding": "FLOAT",
                "denoise": "FLOAT", "steps": "INT", "guidance": "FLOAT",
                "seed": "INT", "guide_size": "INT", "max_targets": "INT",
                "feather": "INT"},
        outputs={"image": "IMAGE", "count": "INT"},
        description="FLUX ADetailer 式局部重绘：YOLO 检测（detector=face/hand，"
                    "模型在 DETECTOR_DIR）→ 裁剪外扩 → 放大到 guide_size → FLUX "
                    "img2img 重绘（FluxImg2ImgPipeline 按组件装配；实际去噪步数"
                    "≈steps×denoise，schnell 建议 steps=8）→ 羽化贴回原图。"
                    "修脸/修手专用，同模型同风格")
    def _op_flux_detail_refine(model, vae, cond, image, detector="face",
                               conf=0.3, padding=0.4, denoise=0.4, steps=8,
                               guidance=0.0, seed=-1, guide_size=512,
                               max_targets=4, feather=16):
        from diffusers import FluxImg2ImgPipeline
        from PIL import Image, ImageDraw, ImageFilter
        if type(model).__name__ != "FluxTransformer2DModel":
            raise ValueError("flux.detail_refine: model 需为 flux.unet_load 输出")
        if not isinstance(cond, dict) or "prompt_embeds" not in cond:
            raise ValueError("flux.detail_refine: cond 需为 flux.encode 输出")
        if not torch.cuda.is_available():
            raise ValueError("flux.detail_refine 需要 CUDA")

        seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**32 - 1)
        steps = int(steps)
        job = execution.current()
        rec = preview.recorder_for(job.id, 1) if job else None

        yolo = _detector(detector)
        results = yolo.predict(image, conf=float(conf), verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            print(f"[flux.detail_refine] {detector}: no target detected",
                  flush=True)
            if job:
                job.set_progress(1, 1)
            return {"image": image, "count": 0}

        xyxy = boxes.xyxy.tolist()
        confs = boxes.conf.tolist()
        order = sorted(range(len(xyxy)), key=lambda i: -confs[i])[:int(max_targets)]

        if rec:
            dbg = image.copy()
            d = ImageDraw.Draw(dbg)
            for i in order:
                x1, y1, x2, y2 = xyxy[i]
                d.rectangle([x1, y1, x2, y2], outline=(34, 211, 238), width=3)
                d.text((x1 + 4, y1 + 4), f"{detector} {confs[i]:.2f}",
                       fill=(34, 211, 238))
            rec.push_pil(_small(dbg), 0.0)

        # 相位：transformer + VAE 上卡（采样相位后来零搬移）
        model.to("cuda")
        vae.to("cuda")
        torch.cuda.empty_cache()
        i2i = FluxImg2ImgPipeline(
            scheduler=_fresh_scheduler(), vae=vae, text_encoder=None,
            tokenizer=None, text_encoder_2=None, tokenizer_2=None,
            transformer=model)
        i2i.set_progress_bar_config(disable=True)

        pe = cond["prompt_embeds"].to("cuda", torch.float16)
        pooled = cond["pooled"].to("cuda", torch.float16)

        img = image.copy()
        W, H = img.size
        n = 0
        total = len(order)
        for i in order:
            execution.check_cancelled()
            x1, y1, x2, y2 = xyxy[i]
            bw, bh = x2 - x1, y2 - y1
            px, py = bw * float(padding), bh * float(padding)
            cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
            cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))
            if cx2 - cx1 < 32 or cy2 - cy1 < 32:
                continue
            crop = img.crop((cx1, cy1, cx2, cy2))
            scale = int(guide_size) / min(crop.size)
            tw = max(64, round(crop.width * scale) // 16 * 16)
            th = max(64, round(crop.height * scale) // 16 * 16)
            crop_big = crop.resize((tw, th), Image.Resampling.LANCZOS)
            t0 = time.time()

            def _cb(p, si, t, kw):
                execution.check_cancelled()
                return kw

            refined = i2i(
                prompt_embeds=pe, pooled_prompt_embeds=pooled,
                image=crop_big, strength=float(denoise),
                num_inference_steps=steps, guidance_scale=float(guidance),
                generator=torch.Generator("cpu").manual_seed(seed + n),
                callback_on_step_end=_cb,
                callback_on_step_end_tensor_inputs=[],
            ).images[0]
            refined = refined.resize(crop.size, Image.Resampling.LANCZOS)
            mask = Image.new("L", crop.size, 0)
            ImageDraw.Draw(mask).rectangle(
                [feather, feather, crop.width - feather,
                 crop.height - feather], fill=255)
            mask = mask.filter(ImageFilter.GaussianBlur(feather / 2))
            img.paste(refined, (cx1, cy1), mask)
            n += 1
            if job:
                job.set_progress(n, total)
            if rec:
                rec.push_pil(_small(img), n / total)
            print(f"[flux.detail_refine] {detector}#{n}/{total} "
                  f"conf={confs[i]:.2f} box=({cx1},{cy1},{cx2},{cy2}) -> "
                  f"{tw}x{th} strength={denoise} ({time.time()-t0:.1f}s)",
                  flush=True)
        return {"image": img, "count": n}
