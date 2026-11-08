"""MiniMax-H3 算子族：联合视频+音频生成（fl2va 首末帧 / t2va 文生），随节点包自注册。

模型事实（对拍 diffusers 0.42.1 内置实现）：
- 管道：MiniMaxH3ModularPipeline（modular only，无 DiffusionPipeline 半子）；
  guidance-distilled——无 negative_prompt、无 cfg，每步一次前向
- 输出：视频帧（24fps 固定，5~15 秒，num_frames 须为 17n+5）+ 立体声波形
  （32kHz，audio_vae 直接 decode 出波形，非 RVQ 码本）
- 文本条件：Qwen3VL-32B 的 hidden_states[50]（fl2va 只走文本塔，vision tower
  仅 ref2va 引用图用——故加载后裁掉 vision tower 省 10GB+ 内存）

权重布局（Comfy-Org int8_convrot 量化包 + 官方 config 目录）：
- /models/minimax-h3/diffusion_models/minimax_h3_fl2va_int8_convrot.safetensors
  transformer int8（**unpruned 版**，官方命名 blocks.N.attn.qkv_proj...），
  加载路径：反量化 → convert_minimax_h3_transformer_checkpoint_to_diffusers
  （diffusers 内置转换：融合 QKV 拆分 + 命名映射）→ 灌入 MiniMaxH3Transformer3DModel。
  ⚠️ 不可用 pruned 版（`*_pruned_int8_convrot.safetensors`）：它用 adaln_t_table
  查找表替换 time_embedder MLP，diffusers 官方转换函数明确拒绝（single_file_utils
  的 ValueError），结构不兼容。
- /models/minimax-h3/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
  Qwen3VL-32B 纯文本塔 int8（model.layers.* 标准 transformers 命名）；
  load_state_dict(strict=False) 跳过 vision tower
- /models/minimax-h3/vae/minimax_h3_video_vae_int8_convrot.safetensors
  视频 VAE int8（官方原始命名）→ _remap_video_vae 映射进 AutoencoderKLMiniMaxH3：
  x_embedder→proj_in / attn.to_qkv 按 dim0 三等分→to_q,to_k,to_v /
  attn.to_out→to_out.0 / ff.w1→ff.net.0.proj / ff.w2→ff.net.2 / 丢弃 mask_token
- /models/minimax-h3/vae/minimax_h3_audio_vae_official.safetensors
  音频 VAE 官方 fp32（diffusers 命名，含 weight_norm 的 weight_g/weight_v，
  Comfy-Org 重打包版去掉了参数化故不用）→ AutoencoderKLMiniMaxH3Audio 直读
- /models/minimax-h3-config/FL2VA/ 官方 config 目录（model_index.json +
  各组件 config.json + tokenizer/processor，KB 级，from_pretrained 的 config 来源）

量化格式（int8_convrot，实测）：量化层 = weight(int8) + weight_scale(fp32,
per-output-channel)，反量化 W = W_int8.float() * scale；comfy_quant 是 ComfyUI
元数据（72B）直接忽略；embedding/norm/bias 与 fp32 pinned 层不量化、原样保留。
transformer 的 _keep_in_fp32_modules（proj_in/audio_proj_in/time_embedder/
proj_out/audio_proj_out/rope）load_state_dict 时由模型侧 dtype 自动 cast 保证。

显存/内存相位（24GB 显存 + 64GB 内存的核心约束——bf16 全量在数学上不成立：
unpruned transformer bf16 ~63GB + TE bf16 ~50GB > 64GB 内存，真机实测 bf16
实例化阶段进程被 OOM 杀。故 transformer/TE 走 **int8 动态推理**（方案 C）：
量化层 _Int8DequantLinear 常驻 int8（transformer ~32GB / TE ~26GB），
forward 时 dequant——与 bf16 全量逐位等价（dequant 是确定映射，量化误差在
Comfy-Org 发布时已固定，ComfyUI 原生即此路径）；
- transformer / text_encoder：int8 常驻 + accelerate cpu_offload_with_hook
  逐层 swap（H3 的 encoders.py 原生兼容 _hf_hook.pre_forward）
- video_vae / audio_vae 小（~5GB / ~0.6GB bf16 直载），随用随 .to(device)
- turbo LoRA（ComfyUI 格式 bf16）经 MiniMaxH3LoraLoaderMixin.load_lora_weights
  挂到 transformer 分区，4 步出片

算子：
- minimax.h3.load(transformer, text_encoder, video_vae, audio_vae, lora,
  lora_scale, dtype, offload) → MODEL（装配好的 MiniMaxH3ModularPipeline，
  进 model_manager LRU，键 minimax-h3:<...>）
- minimax.h3.gen(model, prompt, image?, last_image?, num_frames, seed, steps,
  preview_every) → VIDEO（frames+fps+audio+sample_rate，音轨随 encode_mp4
  mux 进 mp4）+ seed；分钟级任务，请经 POST /jobs 异步执行
"""

OP_VERSION = "2e8d8444afba"
OP_FAMILY = "minimax_h3"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


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

from app import execution, preview

H3_DIR = os.environ.get("MINIMAX_H3_DIR", "/models/minimax-h3")
H3_CONFIG_DIR = os.environ.get("MINIMAX_H3_CONFIG_DIR",
                               "/models/minimax-h3-config/FL2VA")
H3_FPS = 24  # 官方固定帧率（modular_pipeline.MINIMAX_H3_FPS）

# 反量化目标精度：量化层统一 bf16（与官方混合精度 checkpoint 的主体一致）


# ---------------------------------------------------------------------------
# int8 动态推理（方案 C：量化层常驻 int8，forward 时 dequant）
#
# 数值等价性：W = int8.float() * scale 是确定映射——「forward 时 dequant」与
# 「先 dequant 落盘成 bf16 再推理」结果完全一致；量化误差在 Comfy-Org 发布
# int8_convrot 时已固定，与存储格式无关（ComfyUI 原生即此路径）。
#
# 为什么必须动态：unpruned transformer bf16 ~63GB + TE bf16 ~50GB，64GB 内存
# 数学不成立（exec 真机：bf16 实例化阶段进程被 OOM 杀）。int8 常驻后
# transformer ~32GB / TE ~26GB，64GB 可行；配 sequential offload 逐层上卡，
# 显存峰值 = 单层 int8 + 瞬时 bf16 反量化副本（几 GB），24GB 从容。
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- sync-common 内联点：comfy_quant 通用件（分发时拼接，勿手动复制）---
from comfy_quant import (_Int8DequantLinear, _dequant_safetensors,
                         _hadamard, _load_dequant_and_materialize,
                         _materialize_mixed, _rename_scales_like,
                         _rotate_weight_convrot, _sequential_offload,
                         _split_quant_sd)
# --- 内联点结束 ---


# ---------------------------------------------------------------------------
# 反量化 / key 映射（video_vae 小体积组件仍走 bf16 直载路径）
# ---------------------------------------------------------------------------

def _remap_video_vae(sd):
    """Comfy-Org/官方原始命名 → diffusers AutoencoderKLMiniMaxH3 命名。

    decoder 侧：
    - x_embedder.{weight,bias}          → proj_in.{weight,bias}
    - attn.to_qkv.weight (3D,in)        → to_q/to_k/to_v.weight（dim0 三等分，q,k,v 序）
      （bias 同理）
    - attn.to_out.{weight,bias}         → attn.to_out.0.{weight,bias}（ModuleList[0]）
    - ff.w1.{weight,bias}               → ff.net.0.proj.{weight,bias}，**两半交换**：
      官方存 [gate; value]，diffusers SwiGLU 读 [value; gate]（同 transformer 官方
      转换函数 convert_minimax_h3_gated_ff 的处理，对拍 single_file_utils.py）
    - ff.w2.{weight,bias}               → ff.net.2.{weight,bias}
    - decoder.mask_token                → 丢弃（diffusers 重写版无此参数）
    encoder 侧（机械改名）：
    - encoder.down.N.block.M.*          → encoder.down_blocks.N.resnets.M.*
    - encoder.down.N.downsample.conv.*  → encoder.down_blocks.N.downsamplers.0.conv.*
    - nin_shortcut                      → conv_shortcut
    """
    import torch

    out = {}
    for k, t in sd.items():
        if k.endswith("mask_token"):
            continue
        if k in ("latents_mean", "latents_std"):
            # 官方 checkpoint 附带的 buffer 副本；diffusers 从 config.latents_mean/std
            # 读取（decoders.py 的去归一化路径），state_dict 无此 key，丢弃
            continue
        nk = k
        if ".x_embedder." in nk:
            nk = nk.replace(".x_embedder.", ".proj_in.")
        elif nk.endswith(".attn.to_qkv.weight") or nk.endswith(".attn.to_qkv.bias"):
            base, _, kind = nk.rpartition(".attn.to_qkv.")
            q, kk, v = t.chunk(3, dim=0)
            out[f"{base}.attn.to_q.{kind}"] = q.contiguous()
            out[f"{base}.attn.to_k.{kind}"] = kk.contiguous()
            out[f"{base}.attn.to_v.{kind}"] = v.contiguous()
            continue
        elif ".attn.to_out." in nk and ".to_out.0." not in nk:
            nk = nk.replace(".attn.to_out.", ".attn.to_out.0.")
        elif ".ff.w1." in nk:
            # SwiGLU 门控重排：[gate; value] → [value; gate]
            gate, value = t.chunk(2, dim=0)
            t = torch.cat([value, gate], dim=0)
            nk = nk.replace(".ff.w1.", ".ff.net.0.proj.")
        elif ".ff.w2." in nk:
            nk = nk.replace(".ff.w2.", ".ff.net.2.")
        elif nk.startswith("encoder.down."):
            # encoder.down.N.block.M.* → encoder.down_blocks.N.resnets.M.*
            # encoder.down.N.downsample.* → encoder.down_blocks.N.downsamplers.0.*
            nk = nk.replace("encoder.down.", "encoder.down_blocks.", 1)
            nk = nk.replace(".block.", ".resnets.")
            nk = nk.replace(".downsample.", ".downsamplers.0.")
            nk = nk.replace("nin_shortcut", "conv_shortcut")
        out[nk] = t
    return out


# ---------------------------------------------------------------------------
# 组件装配
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
    print(f"[minimax-h3] registered '{key}' est={est_bytes/1e9:.1f}GB", flush=True)
    return obj


def _require_file(path, what):
    if not os.path.isfile(path):
        raise ValueError(f"minimax-h3: 缺 {what}（{path}）")
    return path


def _build_transformer(path, dtype):
    """int8 checkpoint → 官方转换函数（int8 tensor 的 chunk/cat 同样成立）
    → meta 实例化 → 量化层 _Int8DequantLinear 替换（int8 常驻，~32GB）。"""
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.loaders.single_file_utils import (
        convert_minimax_h3_transformer_checkpoint_to_diffusers)

    t0 = time.time()
    cfg = MiniMaxH3Transformer3DModel.load_config(
        os.path.join(H3_CONFIG_DIR, "transformer", "config.json"))
    base, scales, convrot = _split_quant_sd(path)
    print(f"[minimax-h3] transformer 读盘 {len(base)} keys "
          f"（量化 {len(scales)} 层，convrot {len(convrot)} 层）"
          f"{time.time()-t0:.1f}s", flush=True)

    # 官方转换函数对 base+scale 一起跑：scale 的 key 会被同一 RENAME_DICT 改名；
    # 特殊键（.attn.qkv_proj.weight / .ff.net.0.proj.weight）不匹配 *_scale，
    # scale 只改名不变形，QKV 拆分 / gated 交换在下面手动平行处理。
    merged = dict(base)
    merged.update({k + "_scale": s for k, s in scales.items()})
    converted = convert_minimax_h3_transformer_checkpoint_to_diffusers(merged, cfg)
    del base, scales, merged
    scale_sd = {k[:-len("_scale")]: converted.pop(k)
                for k in list(converted.keys()) if k.endswith("_scale")}
    base_sd = converted

    # scale 平行变形 ①：QKV 三分（与 weight 的 chunk(3) 同布局）
    for k in list(scale_sd.keys()):
        if ".attn.qkv_proj.weight" in k:
            s = scale_sd.pop(k)
            q, kk, v = s.chunk(3, dim=0)
            prefix = k.replace(".attn.qkv_proj.weight", "")
            scale_sd[f"{prefix}.attn.to_q.weight"] = q.contiguous()
            scale_sd[f"{prefix}.attn.to_k.weight"] = kk.contiguous()
            scale_sd[f"{prefix}.attn.to_v.weight"] = v.contiguous()
    # scale 平行变形 ②：gated FF 两半交换（[gate;value]→[value;gate]）
    for k in list(scale_sd.keys()):
        if ".ff.net.0.proj.weight" in k:
            s = scale_sd[k]
            gate, value = s.chunk(2, dim=0)
            scale_sd[k] = torch.cat([value, gate], dim=0).contiguous()
    # convrot 平行变形：只改名/拆分（groupsize 是标量，不随数值变形）
    convrot_sd = {}
    for k, g in convrot.items():
        nk = k
        if nk.startswith("blocks."):
            nk = nk.replace("blocks.", "transformer_blocks.", 1)
        for old, new in (("token_refiner.blocks.", "token_refiner.refiner_blocks."),
                         (".attn.out_proj.", ".attn.to_out.0."),
                         (".mlp.fc1.", ".ff.net.0.proj."),
                         (".mlp.fc2.", ".ff.net.2.")):
            nk = nk.replace(old, new)
        if ".attn.qkv_proj.weight" in nk:
            prefix = nk.replace(".attn.qkv_proj.weight", "")
            for sub in ("to_q", "to_k", "to_v"):
                convrot_sd[f"{prefix}.attn.{sub}.weight"] = g
        else:
            convrot_sd[nk] = g
    del convrot
    print(f"[minimax-h3] transformer 转换完成 {time.time()-t0:.1f}s", flush=True)

    with torch.device("meta"):
        model = MiniMaxH3Transformer3DModel.from_config(cfg)
    n_q, missing = _materialize_mixed(model, base_sd, scale_sd, "transformer",
                                      convrot_sd=convrot_sd)
    del base_sd, scale_sd, convrot_sd
    if missing:
        raise ValueError(f"transformer 缺权重 {len(missing)} 项：{missing[:5]}")
    model.eval()
    print(f"[minimax-h3] transformer ready（int8 动态 {n_q} 层）"
          f" {time.time()-t0:.1f}s", flush=True)
    return model


def _attach_lora(model, path, strength=1.0):
    """ComfyUI 格式 turbo LoRA 手动挂载到 _Int8DequantLinear 旁路。

    不走 PEFT / MiniMaxH3LoraLoaderMixin：PEFT 只认 nn.Linear（真机实测
    '_Int8DequantLinear is not supported'），且 ComfyUI LoRA 用**融合层命名**
    （attn.qkv_proj / mlp.fc1），需与官方转换函数同序拆分映射到 diffusers 名：
    qkv 按输出维 chunk(3) → to_q/to_k/to_v；fc1 做 [gate;value]→[value;gate]
    交换 → ff.net.0.proj；fc2 → ff.net.2；out_proj → attn.to_out.0。
    缩放 = alpha / rank × strength。
    """
    import re
    from safetensors import safe_open
    pairs = {}
    with safe_open(path, framework="pt") as f:
        for k in f.keys():
            m = re.match(
                r"diffusion_model\.(.+)\.(lora_A\.weight|lora_B\.weight|alpha)$", k)
            if m:
                pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k)
    if not pairs:
        raise ValueError(f"LoRA 文件无 ComfyUI 格式 key：{path}")

    def targets(orig):
        # ComfyUI 的 blocks.N 对应 diffusers 的 transformer_blocks.N；
        # refiner_blocks.N 对应 token_refiner.refiner_blocks.N（真机实测命名）
        if orig.startswith("blocks."):
            orig = "transformer_" + orig
        elif orig.startswith("refiner_blocks."):
            orig = "token_refiner." + orig
        if orig.endswith(".attn.qkv_proj"):
            p = orig[: -len(".attn.qkv_proj")] + ".attn"
            return [(f"{p}.to_q", 3, 0, False), (f"{p}.to_k", 3, 1, False),
                    (f"{p}.to_v", 3, 2, False)]
        if orig.endswith(".attn.out_proj"):
            return [(orig[: -len(".out_proj")] + ".to_out.0", 1, 0, False)]
        if orig.endswith(".mlp.fc1"):
            return [(orig[: -len(".mlp.fc1")] + ".ff.net.0.proj", 1, 0, True)]
        if orig.endswith(".mlp.fc2"):
            return [(orig[: -len(".mlp.fc2")] + ".ff.net.2", 1, 0, False)]
        return []

    n, skipped = 0, 0
    for orig, tensors in pairs.items():
        A = tensors.get("lora_A.weight")  # (rank, in)
        B = tensors.get("lora_B.weight")  # (out, rank)
        if A is None or B is None:
            continue
        alpha = tensors.get("alpha")
        scale = (float(alpha.flatten()[0]) / A.shape[0]
                 if alpha is not None else 1.0) * strength
        for mod_path, parts, idx, swap in targets(orig):
            try:
                mod = model.get_submodule(mod_path)
            except AttributeError:
                skipped += 1
                continue
            if not isinstance(mod, _Int8DequantLinear):
                skipped += 1
                continue
            Bi = torch.chunk(B, parts, dim=0)[idx] if parts > 1 else B
            if swap:
                half = Bi.shape[0] // 2
                Bi = torch.cat([Bi[half:], Bi[:half]], dim=0)
            mod.add_lora(A, Bi, scale)
            n += 1
    print(f"[minimax-h3] LoRA 挂载 {n} 层（{len(pairs)} 对，跳过 {skipped}）",
          flush=True)
    if n == 0:
        raise ValueError("LoRA 未挂到任何层——key 命名与 transformer 不匹配")


def _build_video_vae(path, dtype):
    """video_vae 加载。

    【重大踩坑】Comfy-Org 的 int8 video_vae 是**自定义几何**（官方 klvae
    AutoencoderKLLegacy，22 帧→6 latent 帧），与 diffusers 内置
    AutoencoderKLMiniMaxH3（7 latent 帧，配套 H3 管道的 clip_length=17
    chunking）**本质不兼容**——key 映射能全对上（missing=0）但 encode
    行为不同，真机实测往返相关仅 0.23（花屏）。必须用 HF 官方仓库的
    **diffusers 版权重**（vae/ 目录 3 分片，from_pretrained 原生加载，
    真机实测往返相关 0.986）。
    path：diffusers 目录（推荐）或 int8 单文件（回退，已知不兼容仅留档）。
    """
    from diffusers import AutoencoderKLMiniMaxH3

    t0 = time.time()
    if os.path.isdir(path):
        import torch
        dt = torch.bfloat16 if dtype == "bf16" else torch.float32
        vae = AutoencoderKLMiniMaxH3.from_pretrained(path, torch_dtype=dt)
        vae.eval()
        print(f"[minimax-h3] video_vae ready（diffusers 原生） "
              f"{time.time()-t0:.1f}s", flush=True)
        return vae
    cfg = AutoencoderKLMiniMaxH3.load_config(
        os.path.join(H3_CONFIG_DIR, "video_vae", "config.json"))
    raw = _dequant_safetensors(path)
    sd = _remap_video_vae(raw)
    del raw
    vae = AutoencoderKLMiniMaxH3.from_config(cfg)
    _load_dequant_and_materialize(vae, sd)
    del sd
    vae.eval()
    print(f"[minimax-h3] video_vae ready {time.time()-t0:.1f}s", flush=True)
    return vae


def _build_audio_vae(path, dtype):
    """官方权重（diffusers 命名，weight_g/weight_v 参数化齐整）直读。"""
    from diffusers import AutoencoderKLMiniMaxH3Audio
    from safetensors.torch import load_file

    t0 = time.time()
    cfg = AutoencoderKLMiniMaxH3Audio.load_config(
        os.path.join(H3_CONFIG_DIR, "audio_vae", "config.json"))
    vae = AutoencoderKLMiniMaxH3Audio.from_config(cfg)
    _load_dequant_and_materialize(vae, load_file(path))
    vae.eval()
    print(f"[minimax-h3] audio_vae ready {time.time()-t0:.1f}s", flush=True)
    return vae


def _build_text_encoder(path, dtype):
    """Qwen3VL-32B 完整加载（含 vision tower）：量化层 int8 动态推理（~26GB 常驻），
    非量化层（embedding/norm/bias）原精度直载。"""
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    t0 = time.time()
    cfg = Qwen3VLConfig.from_pretrained(
        os.path.join(H3_CONFIG_DIR, "text_encoder"))
    base, scales, convrot = _split_quant_sd(path)
    # Comfy-Org 的 TE 是**截断版**（H3 只读 hidden_states[50]，剪掉后 14 层
    # 省 10GB+）：文件文本层 0~49 / visual 层 0~26，而官方 config 是 64 层——
    # 按文件实际层数覆盖 config 再实例化（真机实测缺层 layers.50+ 报错）。
    import re
    n_text = max(int(m.group(1)) for k in base
                 if (m := re.search(r"model\.layers\.(\d+)\.", k))) + 1
    n_vis = max((int(m.group(1)) for k in base
                 if (m := re.search(r"visual\.blocks\.(\d+)\.", k))),
                default=-1) + 1
    if getattr(cfg, "text_config", None) is not None:
        if cfg.text_config.num_hidden_layers != n_text:
            print(f"[minimax-h3] TE 截断版：text layers "
                  f"{cfg.text_config.num_hidden_layers} -> {n_text}", flush=True)
            cfg.text_config.num_hidden_layers = n_text
    if n_vis > 0 and getattr(cfg, "vision_config", None) is not None:
        vc = cfg.vision_config
        for attr in ("depth", "num_hidden_layers"):
            if getattr(vc, attr, None) is not None and getattr(vc, attr) != n_vis:
                print(f"[minimax-h3] TE 截断版：vision {attr} "
                      f"{getattr(vc, attr)} -> {n_vis}", flush=True)
                setattr(vc, attr, n_vis)
    # Comfy-Org 的 TE key 是 Qwen3VLModel 级别命名（visual.* / model.layers.*），
    # ForConditionalGeneration 的参数路径多一层包装（model.visual.* /
    # model.language_model.layers.*）——真机实测映射规则如下：
    def _remap(k):
        if k.startswith("visual."):
            return "model." + k
        if k.startswith("model."):
            return "model.language_" + k
        return k
    base = {_remap(k): v for k, v in base.items()}
    scales = {_remap(k): v for k, v in scales.items()}
    convrot = {_remap(k): g for k, g in convrot.items()}
    with torch.device("meta"):
        # transformers API：直接构造（from_config 是 diffusers 的 ConfigMixin，
        # transformers 模型没有——真机实测 AttributeError）
        te = Qwen3VLForConditionalGeneration(cfg)
    n_q, missing = _materialize_mixed(te, base, scales, "text_encoder",
                                      zero_fill_missing=True,
                                      convrot_sd=convrot)
    del base, scales, convrot
    # 截断版无 final norm——这是 Comfy-Org 的精心设计而非缺失：H3 读
    # hidden_states[50] 需 pre-norm 语义（官方 64 层模型的中间层输出）；
    # transformers 的 hidden_states 元组最后一项是 final norm 后的输出，
    # 若 norm 存在，50 层模型的 hidden_states[50] 就变成 post-norm（语义错）。
    # 置 Identity 跳过后，hidden_states[50] = layer 49 原始输出 ✓
    lm = getattr(getattr(te, "model", None), "language_model", None)
    if lm is not None and getattr(lm, "norm", None) is not None:
        import torch.nn as _nn
        lm.norm = _nn.Identity()
        print("[minimax-h3] TE final norm -> Identity（截断版 pre-norm 语义）",
              flush=True)
    # diffusers 的 get_qwen3vl_prompt_embeds 有硬校验：config 层数必须 >50，
    # 否则拒绝（它担心 50 层模型的 hidden_states[50] 是 post-norm）。但校验读的
    # 是 config 字段而非实际模块树——我们 norm 已是 Identity，hidden_states[50]
    # 恰为第 50 层 pre-norm 输出，与官方 64 层模型的 hidden_states[50] 语义一致
    # （这正是 Comfy-Org 截断版的设计）。把 config 字段改回 51 通过校验。
    if getattr(getattr(te, "config", None), "text_config", None) is not None:
        te.config.text_config.num_hidden_layers = n_text + 1
    # LM head 与 final norm 不随 checkpoint，missing 只许是它们
    bad = [k for k in missing if not k.startswith(("lm_head",))]
    bad = [k for k in bad if "norm" not in k]
    if bad:
        raise ValueError(f"text_encoder 缺层：{bad[:5]}")
    te.eval()
    print(f"[minimax-h3] text_encoder ready（int8 动态 {n_q} 层）"
          f" {time.time()-t0:.1f}s", flush=True)
    return te


def _build_tokenizer_processor():
    """tokenizer + processor（官方 config 目录直读）。"""
    from transformers import Qwen2TokenizerFast, Qwen3VLProcessor

    tok = Qwen2TokenizerFast.from_pretrained(
        os.path.join(H3_CONFIG_DIR, "tokenizer"))
    proc = Qwen3VLProcessor.from_pretrained(
        os.path.join(H3_CONFIG_DIR, "processor"))
    return tok, proc


def _build_schedulers():
    """双 scheduler：video shift=12.0 / audio shift=3.0（官方契约）。"""
    from diffusers import MiniMaxH3Scheduler

    sched = MiniMaxH3Scheduler.from_config(
        os.path.join(H3_CONFIG_DIR, "scheduler", "scheduler_config.json")) \
        if os.path.isfile(os.path.join(
            H3_CONFIG_DIR, "scheduler", "scheduler_config.json")) \
        else MiniMaxH3Scheduler()
    audio_sched = MiniMaxH3Scheduler()
    return sched, audio_sched


def _build_references(references_json):
    """references JSON → 官方 Reference 对象列表（ref2va 的输入构造）。

    格式：[{"type": "image"|"video"|"audio", "id": "<对象仓库 id>"}]
    顺序即语义（prompt 里的 <Picture i>/<Video k>/<Audio j> 标注按此编号）。
    上限：9 图 + 3 视频 + 3 音频 = 12（官方 checkpoint 文档值，setup step
    会再校验一遍）。音频参考不能单独出现。
    """
    import json as _json

    from diffusers.modular_pipelines.minimax_h3.references import (
        MiniMaxH3AudioReference, MiniMaxH3ImageReference,
        MiniMaxH3VideoReference)

    try:
        spec = _json.loads(references_json)
    except ValueError as e:
        raise ValueError(f"minimax.h3.gen: references JSON 解析失败: {e}")
    if not isinstance(spec, list) or not spec:
        raise ValueError("minimax.h3.gen: references 须为非空 JSON 数组")

    from app.main import store  # 延迟导入：插件注册时 main 未必就绪

    refs = []
    for i, item in enumerate(spec):
        if not isinstance(item, dict):
            raise ValueError(f"references[{i}] 须为对象 {{'type','id'}}")
        rtype, rid = item.get("type"), item.get("id")
        try:
            if rtype == "image":
                data = store.get(rid, "IMAGE")["data"]
                pil = data if not isinstance(data, list) else data[0]
                refs.append(MiniMaxH3ImageReference(image=pil))
            elif rtype == "video":
                v = store.get(rid, "VIDEO")["data"]
                # gen 产物的内存形态：frames(PIL 列表)/fps/audio/sample_rate
                refs.append(MiniMaxH3VideoReference(
                    frames=v["frames"], fps=float(v.get("fps") or 24.0),
                    audio=v.get("audio"),
                    sample_rate=v.get("sample_rate")))
            elif rtype == "audio":
                a = store.get(rid, "AUDIO")["data"]
                refs.append(MiniMaxH3AudioReference(
                    audio=a["audio"], sample_rate=a.get("sample_rate")))
            else:
                raise ValueError(
                    f"references[{i}].type 须为 image/video/audio，"
                    f"收到 {rtype!r}")
        except (KeyError, TypeError) as e:
            raise ValueError(f"references[{i}] 对象不存在或类型不对: {e}")
    n_img = sum(1 for r in refs if r.kind == "image")
    n_vid = sum(1 for r in refs if r.kind == "video")
    n_aud = sum(1 for r in refs if r.kind == "audio")
    print(f"[minimax-h3] references: {n_img} 图 + {n_vid} 视频 + "
          f"{n_aud} 音频（上限 9/3/3）", flush=True)
    return refs


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "minimax.h3.load",
        inputs={"transformer": "STRING", "text_encoder": "STRING",
                "video_vae": "STRING", "audio_vae": "STRING",
                "lora": "STRING", "lora_scale": "FLOAT",
                "dtype": "STRING", "offload": "STRING"},
        outputs={"model": "MODEL"},
        description="MiniMax-H3 装配：Comfy-Org int8 权重反量化 + 官方 config → "
                    "MiniMaxH3ModularPipeline（transformer/TE 逐层 offload，"
                    "VAE 随用上卡）。可选 turbo LoRA（4 步）。进 model_manager LRU")
    def _op_load(transformer="minimax_h3_fl2va_int8_convrot.safetensors",
                 text_encoder="qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
                 video_vae="vae-diffusers",
                 audio_vae="minimax_h3_audio_vae_official.safetensors",
                 lora="", lora_scale=1.0, dtype="bf16", offload="sequential"):
        import torch
        from diffusers import MiniMaxH3ModularPipeline

        key = (f"minimax-h3:{transformer}|{text_encoder}|{lora}@{lora_scale}"
               f"|{dtype}|{offload}")
        models = _models()
        if key in models._pipes:
            return {"model": models.get(key)}

        _require_file(os.path.join(H3_CONFIG_DIR, "model_index.json"),
                      "官方 config 目录")
        tr_path = _require_file(
            os.path.join(H3_DIR, "diffusion_models", transformer), "transformer")
        te_path = _require_file(
            os.path.join(H3_DIR, "text_encoders", text_encoder), "text_encoder")
        # video_vae 优先 diffusers 目录（原生权重；int8 文件是不兼容回退）
        vae_path = os.path.join(H3_DIR, video_vae)
        if not os.path.isdir(vae_path):
            vae_path = _require_file(
                os.path.join(H3_DIR, "vae", video_vae), "video_vae")
        avae_path = _require_file(
            os.path.join(H3_DIR, "vae", audio_vae), "audio_vae")

        t0 = time.time()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ① 组件装配（transformer/TE 内存常驻 + offload hook 管上下卡；
        # VAE 体积小直接常驻显存——留 CPU 会让 decode 慢一个数量级）
        transformer_m = _build_transformer(tr_path, dtype)
        text_encoder_m = _build_text_encoder(te_path, dtype)
        video_vae_m = _build_video_vae(vae_path, dtype)
        audio_vae_m = _build_audio_vae(avae_path, dtype)
        tokenizer, processor = _build_tokenizer_processor()
        scheduler, audio_scheduler = _build_schedulers()

        # ② LoRA 必须在 offload arm 之前挂：buffer 注册后 hook 才能正确
        # 统计并搬运（真机顺序坑）。
        if lora:
            lora_path = _require_file(
                os.path.join(H3_DIR, "loras", lora), "LoRA")
            _attach_lora(transformer_m, lora_path, float(lora_scale))

        # ③ VAE 上卡常驻（小：video ~5GB bf16 + audio ~0.5GB）
        if device.type == "cuda":
            video_vae_m = video_vae_m.to(device)
            audio_vae_m = audio_vae_m.to(device)

        # ④ 管道装配（ModularPipeline API 是 register_components——
        # DiffusionPipeline 的 register_modules 它没有，真机实测 AttributeError）
        # ref2va 权重挂 transformer_ref 组件位（与 transformer config 同构，
        # 官方一个仓库两个 checkpoint 分区，workflow 各载各的）
        is_ref = "ref2va" in transformer
        pipe = MiniMaxH3ModularPipeline()
        pipe.register_components(
            text_encoder=text_encoder_m,
            vae=video_vae_m,
            audio_vae=audio_vae_m,
            tokenizer=tokenizer,
            processor=processor,
            scheduler=scheduler,
            audio_scheduler=audio_scheduler,
            **({"transformer_ref": transformer_m} if is_ref
               else {"transformer": transformer_m}),
        )

        # ⑤ offload：大组件逐层 swap（24GB 卡跑 bf16 的必由之路）
        if offload == "sequential" and device.type == "cuda":
            comp = pipe.transformer_ref if is_ref else pipe.transformer
            comp = _sequential_offload(comp, device, "transformer")
            if is_ref:
                pipe.transformer_ref = comp
            else:
                pipe.transformer = comp
            pipe.text_encoder = _sequential_offload(
                pipe.text_encoder, device, "text_encoder")

        # ⑥ _execution_device sentinel：diffusers 靠组件的
        # _hf_hook.execution_device 推断执行设备（真机读源码确认）；
        # 没有 _hf_hook 的组件会回落 pipe.device=cpu → input_ids 留在 cpu
        # 撞上已上卡的 embedding（index_select 设备冲突）。挂 sentinel 指明
        # cuda；无 pre_forward 属性，encoders.py 的手动 hook 触发安全跳过。
        if device.type == "cuda":
            class _ExecDeviceSentinel:
                execution_device = "cuda"
            tr = pipe.transformer_ref if is_ref else pipe.transformer
            tr._hf_hook = _ExecDeviceSentinel()
            pipe.text_encoder._hf_hook = _ExecDeviceSentinel()

        def _sz(p):
            if os.path.isdir(p):
                return sum(os.path.getsize(os.path.join(dp, f))
                           for dp, _, fs in os.walk(p) for f in fs)
            return os.path.getsize(p)
        est = sum(_sz(p) for p in (tr_path, te_path, vae_path, avae_path))
        print(f"[minimax-h3] pipeline assembled {time.time()-t0:.1f}s", flush=True)
        return {"model": _register(key, pipe, est, "MiniMaxH3ModularPipeline")}

    @registry.register(
        "minimax.h3.gen",
        inputs={"model": "MODEL", "prompt": "STRING",
                "image": "IMAGE", "last_image": "IMAGE",
                "references": "STRING",
                "num_frames": "INT", "seed": "INT", "steps": "INT",
                "preview_every": "INT"},
        outputs={"video": "VIDEO", "seed": "INT"},
        description="MiniMax-H3 生成：fl2va（首/末帧）或 t2va（纯文本）或 "
                    "ref2va（references JSON：图/视频/音频参考列表，上限 "
                    "9图+3视频+3音频，prompt 里用 <Picture 1> 等点名）→ "
                    "联合视频+音频。guidance-distilled 无负提示/cfg；24fps 固定，"
                    "num_frames 自动对齐 17n+5（5~15s）。输出 VIDEO 含 32kHz "
                    "立体声音轨（encode_mp4 时 mux 进 mp4）。分钟级任务，"
                    "请经 POST /jobs 异步执行")
    def _op_gen(model, prompt="", image=None, last_image=None, num_frames=125,
                seed=-1, steps=4, preview_every=0, references=""):
        import random

        import torch
        from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
            align_num_frames)

        pipe = model
        if not prompt:
            raise ValueError("minimax.h3.gen: prompt 必填（H3 是强文本引导模型）")

        # ref2va：references JSON 列表 [{'type':'image|video|audio','id':'<对象id>'}]
        # —— 与 image/last_image 互斥（官方 Auto blocks 的分流优先级是
        # references > keyframes > 纯文本）
        refs = _build_references(references) if references else None
        if refs and (image is not None or last_image is not None):
            raise ValueError(
                "minimax.h3.gen: references 与 image/last_image 互斥"
                "（ref2va 与 fl2va 是两个 checkpoint 分区）")
        if refs and getattr(pipe, "transformer_ref", None) is None:
            raise ValueError(
                "minimax.h3.gen: 当前模型不是 ref2va 装配"
                "（load 时 transformer 需选 *_ref2va_* 权重）")

        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        # 帧数对齐（17n+5，视频 VAE 的 chunk 契约；5~15 秒 = 125~365 帧）
        aligned = align_num_frames(int(num_frames), 17, 5)
        if aligned != num_frames:
            print(f"[minimax-h3] num_frames {num_frames} -> {aligned}（17n+5 对齐）",
                  flush=True)
        secs = aligned / H3_FPS
        if not (pipe.min_duration - 1e-6 <= secs <= pipe.max_duration + 1e-6):
            raise ValueError(
                f"minimax.h3.gen: 时长 {secs:.1f}s 超出 H3 范围 "
                f"[{pipe.min_duration:g}, {pipe.max_duration:g}] 秒")

        kwargs = {"prompt": prompt, "num_frames": aligned,
                  "num_inference_steps": int(steps), "generator": gen}
        if refs:
            kwargs["references"] = refs
        if image is not None:
            kwargs["image"] = image
        if last_image is not None:
            kwargs["last_image"] = last_image

        job = execution.current()
        if job is not None:
            job.set_progress(0, 3)  # 粗粒度阶段：encode/denoise/decode

        t0 = time.time()
        if execution.check_cancelled is not None:
            execution.check_cancelled()
        out = pipe(output=["videos", "audio", "sampling_rate"], **kwargs)
        if execution.check_cancelled is not None:
            execution.check_cancelled()

        videos = out["videos"]
        audio = out.get("audio")
        sr = int(out.get("sampling_rate") or 32000)
        frames = videos[0] if isinstance(videos, (list, tuple)) and videos and \
            isinstance(videos[0], (list, tuple)) else videos
        if job is not None:
            job.set_progress(3, 3)

        video_obj = {"frames": list(frames), "fps": H3_FPS}
        if audio is not None:
            video_obj["audio"] = audio
            video_obj["sample_rate"] = sr
        print(f"[minimax-h3] gen done in {time.time()-t0:.1f}s "
              f"frames={len(video_obj['frames'])} audio="
              f"{'yes' if audio is not None else 'no'} seed={seed}", flush=True)
        return {"video": video_obj, "seed": seed}
