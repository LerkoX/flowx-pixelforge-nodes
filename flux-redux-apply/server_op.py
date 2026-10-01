"""FLUX Redux 参考图注入服务端算子插件（对标 ComfyUI Load Style Model +
Apply Style Model 的 Redux 路径，随节点包自注册）。

机制（对拍 ComfyUI comfy/ldm/flux/redux.py + StyleModelApply）：
- SigLIP-so400m-patch14-384 编码参考图 → last_hidden_state [1, 729, 1152]
- Redux 适配器（flux1-redux-dev.safetensors）：Linear(1152→4096*3) → SiLU
  → Linear(4096*3→4096)，投影为 [1, 729, 4096]
- strength 乘到投影 embeds 上（ComfyUI multiply 模式），沿序列维拼到文本
  prompt_embeds 前面：cat([redux, text], dim=1)，pooled 保持不变
- 采样侧零改动：flux.sample 只认 prompt_embeds/pooled，序列变长自然生效

模型布局（服务端文件，需预先放置）：
- SigLIP 视觉编码器：CLIP_VISION_DIR（默认 /models/clip_vision）下的
  transformers 格式目录，如 siglip-so400m-patch14-384/
  （config.json + model.safetensors + preprocessor_config.json，
  只需 vision tower，不要 text tower）
- Redux 适配器：FLUX_DIR（默认 /models/flux）下的
  flux1-redux-dev.safetensors（~129MB）

内存相位（GTX1080 8GB 约束，与 flux.encode 同策略）：
- 先把所有 FluxTransformer2DModel 搬回 CPU 腾 VRAM
- SigLIP(~0.9GB fp16)+适配器(~0.13GB fp16) 上卡编码
- 完成后搬回 CPU 驻留 LRU（release=1 时直接淘汰出 LRU）

算子：
- flux.redux_apply(cond, image, strength, reducer, vision, release)
  → COND{prompt_embeds, pooled}
"""
import os
import time

import torch

FLUX_DIR = os.environ.get("FLUX_DIR", "/models/flux")
CLIP_VISION_DIR = os.environ.get("CLIP_VISION_DIR", "/models/clip_vision")


# ---------------------------------------------------------------------------
# Redux 适配器结构（逐字节对齐 ComfyUI ReduxImageEncoder）
# ---------------------------------------------------------------------------


class ReduxImageEncoder(torch.nn.Module):
    def __init__(self, redux_dim=1152, txt_in_features=4096):
        super().__init__()
        self.redux_up = torch.nn.Linear(redux_dim, txt_in_features * 3)
        self.redux_down = torch.nn.Linear(txt_in_features * 3, txt_in_features)

    def forward(self, sigclip_embeds):
        return self.redux_down(
            torch.nn.functional.silu(self.redux_up(sigclip_embeds)))


# ---------------------------------------------------------------------------
# 组件注册（model_manager LRU；与 flux 细分链同一淘汰/pin/unload 语义）
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
    print(f"[flux-redux] registered '{key}' est={est_bytes/1e9:.2f}GB",
          flush=True)
    return obj


def _evict(key):
    models = _models()
    with models._lock:
        obj = models._pipes.pop(key, None)
        models._archs.pop(key, None)
        models._sizes.pop(key, None)
        models._last_used.pop(key, None)
    if obj is not None:
        del obj
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[flux-redux] evicted '{key}'", flush=True)


def _move_transformers(device):
    """把所有已注册 FluxTransformer2DModel 搬到指定设备（编码相位腾 VRAM）。"""
    n = 0
    for p in list(_models()._pipes.values()):
        if type(p).__name__ == "FluxTransformer2DModel":
            p.to(device)
            n += 1
    if n:
        torch.cuda.empty_cache()
        print(f"[flux-redux] {n} transformer -> {device}", flush=True)


# ---------------------------------------------------------------------------
# 组件装载（CPU 装载进 LRU，编码相位才上卡）
# ---------------------------------------------------------------------------


def _load_vision(name):
    """SigLIP 视觉编码器（transformers 格式目录，fp16 CPU 常驻）。"""
    key = f"flux-redux-vision:{name}"
    models = _models()
    with models._lock:
        hit = models._pipes.get(key)
    if hit is not None:
        return hit
    from transformers import SiglipImageProcessor, SiglipVisionModel
    d = os.path.join(CLIP_VISION_DIR, name)
    if not os.path.isdir(d):
        available = (sorted(os.listdir(CLIP_VISION_DIR))
                     if os.path.isdir(CLIP_VISION_DIR) else [])
        raise FileNotFoundError(
            f"flux.redux_apply: 缺 SigLIP 视觉编码器目录 {d}；"
            f"{CLIP_VISION_DIR} 现有: {available or '(empty)'}。"
            f"请放置 siglip-so400m-patch14-384 的 transformers 格式目录"
            f"（vision tower 即可）")
    t0 = time.time()
    model = SiglipVisionModel.from_pretrained(d, torch_dtype=torch.float16)
    model.eval()
    proc = SiglipImageProcessor.from_pretrained(d)
    bundle = {"model": model, "processor": proc}
    est = sum(p.numel() * 2 for p in model.parameters())
    print(f"[flux-redux] siglip loaded {time.time()-t0:.1f}s "
          f"est={est/1e9:.2f}GB", flush=True)
    _register(key, bundle, est, "SiglipVisionModel(fp16)")
    return bundle


def _load_reducer(name):
    """Redux 投影适配器（safetensors 单文件，fp16 CPU 常驻）。"""
    key = f"flux-redux-adapter:{name}"
    models = _models()
    with models._lock:
        hit = models._pipes.get(key)
    if hit is not None:
        return hit
    from safetensors.torch import load_file
    path = os.path.join(FLUX_DIR, name)
    if not os.path.isfile(path):
        available = (sorted(os.listdir(FLUX_DIR))
                     if os.path.isdir(FLUX_DIR) else [])
        raise FileNotFoundError(
            f"flux.redux_apply: 缺 Redux 适配器 {path}；"
            f"{FLUX_DIR} 现有: {available or '(empty)'}。"
            f"请放置 flux1-redux-dev.safetensors")
    t0 = time.time()
    sd = load_file(path)
    if "redux_down.weight" not in sd:
        raise ValueError(
            f"flux.redux_apply: {path} 不是 Redux 适配器"
            f"（缺 redux_down.weight，keys={sorted(sd)[:6]}...）")
    model = ReduxImageEncoder()
    model.load_state_dict(sd)
    del sd
    model = model.to(torch.float16).eval()
    est = sum(p.numel() * 2 for p in model.parameters())
    print(f"[flux-redux] reducer loaded {time.time()-t0:.1f}s "
          f"est={est/1e6:.0f}MB", flush=True)
    _register(key, model, est, "ReduxImageEncoder(fp16)")
    return model


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------


def register(registry):
    @registry.register(
        "flux.redux_apply",
        inputs={"cond": "COND", "image": "IMAGE", "strength": "FLOAT",
                "reducer": "STRING", "vision": "STRING", "release": "INT"},
        outputs={"cond": "COND", "info": "STRING"},
        description="FLUX Redux 参考图注入（对标 ComfyUI Load Style Model + "
                    "Apply Style Model）：SigLIP 编码参考图 → Redux 适配器投影 "
                    "729 token → 拼到文本 COND 序列前（strength 乘性缩放，"
                    "pooled 不变）。输出接 flux-sampler 的 cond。"
                    "编码相位：transformer 搬回 CPU → SigLIP+适配器上卡 → "
                    "编码 → 回 CPU 驻留（release=1 直接淘汰）")
    def _op_flux_redux_apply(cond, image, strength=1.0,
                             reducer="flux1-redux-dev.safetensors",
                             vision="siglip-so400m-patch14-384", release=0):
        if not (isinstance(cond, dict) and "prompt_embeds" in cond):
            raise ValueError("flux.redux_apply: cond 需为 flux.encode 的输出")
        if not torch.cuda.is_available():
            raise ValueError("flux.redux_apply 需要 CUDA（SigLIP 编码上卡）")
        t0 = time.time()
        # 相位切换：transformer 回 CPU 腾 VRAM，SigLIP+适配器上卡
        _move_transformers("cpu")
        vb = _load_vision(vision)
        rd = _load_reducer(reducer)
        vb["model"].to("cuda")
        rd.to("cuda")

        images = image if isinstance(image, list) else [image]
        px = vb["processor"](images=images, return_tensors="pt") \
            .pixel_values.to("cuda", torch.float16)
        with torch.no_grad():
            hidden = vb["model"](pixel_values=px).last_hidden_state
            proj = rd(hidden) * float(strength)

        pe = cond["prompt_embeds"]
        new_pe = torch.cat([proj.float().cpu(), pe.float().cpu()], dim=1)
        pooled = cond["pooled"]

        vb["model"].to("cpu")
        rd.to("cpu")
        if int(release):
            _evict(f"flux-redux-vision:{vision}")
            _evict(f"flux-redux-adapter:{reducer}")
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        info = (f"{time.time()-t0:.1f}s siglip={tuple(hidden.shape)} "
                f"redux={tuple(proj.shape)} strength={strength} "
                f"pe {tuple(pe.shape)} -> {tuple(new_pe.shape)}")
        print(f"[flux.redux_apply] {info}", flush=True)
        return {"cond": {"prompt_embeds": new_pe, "pooled": pooled},
                "info": info}
