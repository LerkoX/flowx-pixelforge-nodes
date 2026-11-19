"""BrushNet / PowerPaint 算子族（SD1.x 局部重绘，纯插件层零框架改动）。

BrushNet（TencentARC，ECCV'24）：可插拔 inpaint 双分支——UNet 完整副本
（down+mid+up）吃 [noisy_latent(4ch) | vae.encode(masked_image)(4ch) |
mask(1ch)] = 9ch 输入，逐 block 经零卷积输出残差，注入主 UNet 对应位置。
**官方注入是块内注入**（conv_in 后 +1、每个 down block 的每个 resnet/
downsampler 后各 +1、mid 后 +1、每个 up block 的每个 resnet/upsampler 后
各 +1），与 ControlNet 的块间注入（down_block_additional_residuals）
语义不同——本族用 forward hook 在主 UNet 对应子模块输出上做加法精确复刻
（fork 源码 unet_2d_condition.py 逐行对齐），且与 ControlNet kwargs 注入
正交可叠加。

PowerPaint（open-mmlab，ECCV'24）：task prompt 机制——tokenizer 加 3 个
占位符 P_ctxt/P_shape/P_obj（每个 10 向量），学习到的 embedding 在其微调
text_encoder 尾部 30 行；部署时抽成 /models/task_prompts/powerpaint.safetensors
（单 key "embeddings" [30,768]，行序 = P_ctxt×10, P_shape×10, P_obj×10，
与官方 add_tokens 顺序一致）。v2(-1) 是 BrushNet 架构（保留 cross-attention），
走 sd.brushnet.* 链路；v1 是 9ch inpaint UNet，走既有
sd.checkpoint.load_inpaint + sd.sample_inpaint 链路 + 本族 sd.clip.encode_task。

diffusers 官方从未合入 BrushNet（PR #9125 closed unmerged），故本族 vendor
BrushNetModel（源自 TencentARC/BrushNet fork 的 models/brushnet.py，931 行），
对标准 diffusers（0.35.2 真机验证基线）的两处适配：
  1) fork 魔改 UpBlock 的 return_res_samples → 标准块用 hook 捕获各
     resnet/upsampler 输出（触发顺序即官方残差顺序）；
  2) fork 的 MidBlock2D（2 个 ResnetBlock2D、无注意力，键名 resnets.0/1.*）
     → 内置小类（标准 diffusers 的 get_mid_block 无此类型）。

权重布局（MODELS_DIR 下，environ MODELS_DIR 缺省 /models）：
  brushnet/<name>/{config.json?, diffusion_pytorch_model(.fp16)?.safetensors}
    —— Sanster/brushnet_random_mask（diffusers 布局，config._class_name=
    BrushNetModel）；无 config.json 时按 state_dict 键嗅探结构
    （块数/通道/cross-attn/mid 类型），适配 PowerPaint_Brushnet 裸权重。
  task_prompts/powerpaint.safetensors —— task prompt embedding 抽取件。

真机基线：GTX 1080 8GB / torch 2.3.1 / diffusers 0.35.2（pascal 栈）。
"""
import os
import time

import torch
import torch.nn.functional as F
from torch import nn

from app import execution, ops, preview

# --- F1 族版本化：算子身份 = 实现内容 --------------------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
OP_VERSION = "22156ecff34c"
OP_FAMILY = "brushnet"  # 落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs,
                                    outputs, description)
# ---------------------------------------------------------------------------


def _zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class _MidBlock2D(nn.Module):
    """官方 fork 的 MidBlock2D：2 个 ResnetBlock2D、无注意力。

    键名 resnets.0/resnets.1.*（与发布权重逐键对应）；标准 diffusers 的
    UNetMidBlock2D 带 attentions 键，不能直接替。
    """

    def __init__(self, in_channels, temb_channels, dropout=0.0,
                 resnet_eps=1e-6, resnet_time_scale_shift="default",
                 resnet_act_fn="swish", resnet_groups=32,
                 output_scale_factor=1.0, **_ignored):
        super().__init__()
        from diffusers.models.resnet import ResnetBlock2D
        self.has_cross_attention = False
        groups = resnet_groups if resnet_groups is not None \
            else min(in_channels // 4, 32)
        self.resnets = nn.ModuleList([
            ResnetBlock2D(
                in_channels=in_channels, out_channels=in_channels,
                temb_channels=temb_channels, eps=resnet_eps, groups=groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor, pre_norm=True)
            for _ in range(2)])
        self.gradient_checkpointing = False

    def forward(self, hidden_states, temb=None, **_kw):
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb)
        return hidden_states


def _get_mid_block(mid_block_type, temb_channels, in_channels, resnet_eps,
                   resnet_act_fn, resnet_groups, output_scale_factor=1.0,
                   resnet_time_scale_shift="default", cross_attention_dim=None,
                   num_attention_heads=None, use_linear_projection=False,
                   upcast_attention=False):
    """mid_block_type 路由：fork 的 MidBlock2D 内置，其余委托标准 diffusers。"""
    if mid_block_type in (None, "None"):
        return None
    if mid_block_type == "MidBlock2D":
        return _MidBlock2D(
            in_channels=in_channels, temb_channels=temb_channels,
            resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups, output_scale_factor=output_scale_factor,
            resnet_time_scale_shift=resnet_time_scale_shift)
    from diffusers.models.unets.unet_2d_blocks import get_mid_block
    return get_mid_block(
        mid_block_type, temb_channels=temb_channels, in_channels=in_channels,
        resnet_eps=resnet_eps, resnet_act_fn=resnet_act_fn,
        resnet_groups=resnet_groups, output_scale_factor=output_scale_factor,
        transformer_layers_per_block=1,
        num_attention_heads=num_attention_heads,
        cross_attention_dim=cross_attention_dim,
        use_linear_projection=use_linear_projection,
        upcast_attention=upcast_attention,
        resnet_time_scale_shift=resnet_time_scale_shift)


class BrushNetModel(nn.Module):
    """vendor 自官方 fork 的 BrushNetModel（推理必需子集）。

    与官方差异：不继承 ModelMixin/ConfigMixin（避免加载路径分歧，配置由
    _brushnet_config 显式构造）、up 残差用 hook 捕获（见 forward 注释）、
    砍掉 SDXL added_cond/class_embed/attention-processor 管理（v1 只支持
    SD1.x——官方发布的 brushnet 权重也只有 SD1.5）。
    """

    def __init__(self, config):
        super().__init__()
        from diffusers.models.embeddings import (TimestepEmbedding,
                                                 Timesteps)
        from diffusers.models.unets.unet_2d_blocks import (get_down_block,
                                                           get_up_block)
        self.config = dict(config)
        c = self.config
        block_out = tuple(c["block_out_channels"])
        layers_per_block = c.get("layers_per_block", 2)
        time_embed_dim = block_out[0] * 4

        conv_in_kernel = 3
        self.conv_in_condition = nn.Conv2d(
            c["in_channels"] + c["conditioning_channels"], block_out[0],
            kernel_size=conv_in_kernel, padding=(conv_in_kernel - 1) // 2)

        self.time_proj = Timesteps(block_out[0], c.get("flip_sin_to_cos", True),
                                   c.get("freq_shift", 0))
        self.time_embedding = TimestepEmbedding(block_out[0], time_embed_dim,
                                                act_fn=c.get("act_fn", "silu"))

        tlpb = c.get("transformer_layers_per_block", 1)
        if isinstance(tlpb, int):
            tlpb = [tlpb] * len(block_out)
        ahd = c.get("attention_head_dim", 8)
        if isinstance(ahd, int):
            ahd = (ahd,) * len(block_out)
        nah = c.get("num_attention_heads", None) or ahd
        if isinstance(nah, int):
            nah = (nah,) * len(block_out)
        oca = c.get("only_cross_attention", False)
        if isinstance(oca, bool):
            oca = [oca] * len(block_out)

        self.down_blocks = nn.ModuleList([])
        self.brushnet_down_blocks = nn.ModuleList([
            _zero_module(nn.Conv2d(block_out[0], block_out[0], kernel_size=1))])

        output_channel = block_out[0]
        for i, down_block_type in enumerate(c["down_block_types"]):
            input_channel = output_channel
            output_channel = block_out[i]
            is_final_block = i == len(block_out) - 1
            down_block = get_down_block(
                down_block_type,
                num_layers=layers_per_block,
                transformer_layers_per_block=tlpb[i],
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                add_downsample=not is_final_block,
                resnet_eps=c.get("norm_eps", 1e-5),
                resnet_act_fn=c.get("act_fn", "silu"),
                resnet_groups=c.get("norm_num_groups", 32),
                cross_attention_dim=c.get("cross_attention_dim"),
                num_attention_heads=nah[i],
                attention_head_dim=ahd[i] if ahd[i] is not None
                else output_channel,
                downsample_padding=c.get("downsample_padding", 1),
                use_linear_projection=c.get("use_linear_projection", False),
                only_cross_attention=oca[i],
                upcast_attention=c.get("upcast_attention", False),
                resnet_time_scale_shift=c.get("resnet_time_scale_shift",
                                              "default"),
            )
            self.down_blocks.append(down_block)
            for _ in range(layers_per_block):
                self.brushnet_down_blocks.append(_zero_module(
                    nn.Conv2d(output_channel, output_channel, kernel_size=1)))
            if not is_final_block:
                self.brushnet_down_blocks.append(_zero_module(
                    nn.Conv2d(output_channel, output_channel, kernel_size=1)))

        mid_channel = block_out[-1]
        self.brushnet_mid_block = _zero_module(
            nn.Conv2d(mid_channel, mid_channel, kernel_size=1))
        self.mid_block = _get_mid_block(
            c.get("mid_block_type", "MidBlock2D"),
            temb_channels=time_embed_dim,
            in_channels=mid_channel,
            resnet_eps=c.get("norm_eps", 1e-5),
            resnet_act_fn=c.get("act_fn", "silu"),
            resnet_groups=c.get("norm_num_groups", 32),
            output_scale_factor=c.get("mid_block_scale_factor", 1),
            resnet_time_scale_shift=c.get("resnet_time_scale_shift", "default"),
            cross_attention_dim=c.get("cross_attention_dim"),
            num_attention_heads=nah[-1],
            use_linear_projection=c.get("use_linear_projection", False),
            upcast_attention=c.get("upcast_attention", False))

        self.up_blocks = nn.ModuleList([])
        self.brushnet_up_blocks = nn.ModuleList([])
        rev_block_out = list(reversed(block_out))
        rev_nah = list(reversed(nah))
        rev_ahd = list(reversed(ahd))
        rev_tlpb = list(reversed(tlpb))
        rev_oca = list(reversed(oca))
        output_channel = rev_block_out[0]
        for i, up_block_type in enumerate(c["up_block_types"]):
            is_final_block = i == len(block_out) - 1
            prev_output_channel = output_channel
            output_channel = rev_block_out[i]
            input_channel = rev_block_out[min(i + 1, len(block_out) - 1)]
            add_upsample = not is_final_block
            up_block = get_up_block(
                up_block_type,
                num_layers=layers_per_block + 1,
                transformer_layers_per_block=rev_tlpb[i],
                in_channels=input_channel,
                out_channels=output_channel,
                prev_output_channel=prev_output_channel,
                temb_channels=time_embed_dim,
                add_upsample=add_upsample,
                resnet_eps=c.get("norm_eps", 1e-5),
                resnet_act_fn=c.get("act_fn", "silu"),
                resolution_idx=i,
                resnet_groups=c.get("norm_num_groups", 32),
                cross_attention_dim=c.get("cross_attention_dim"),
                num_attention_heads=rev_nah[i],
                attention_head_dim=rev_ahd[i] if rev_ahd[i] is not None
                else output_channel,
                use_linear_projection=c.get("use_linear_projection", False),
                only_cross_attention=rev_oca[i],
                upcast_attention=c.get("upcast_attention", False),
                resnet_time_scale_shift=c.get("resnet_time_scale_shift",
                                              "default"),
            )
            self.up_blocks.append(up_block)
            for _ in range(layers_per_block + 1):
                self.brushnet_up_blocks.append(_zero_module(
                    nn.Conv2d(output_channel, output_channel, kernel_size=1)))
            if not is_final_block:
                self.brushnet_up_blocks.append(_zero_module(
                    nn.Conv2d(output_channel, output_channel, kernel_size=1)))

    @staticmethod
    def _capture_up_res(block, call):
        """替代 fork 的 return_res_samples：hook 捕获 up block 内各
        resnet/upsampler 的输出（触发顺序即官方残差收集顺序）。"""
        outs = []
        handles = []

        def _hook(_m, _a, o):
            outs.append(o if torch.is_tensor(o) else o[0])

        for r in block.resnets:
            handles.append(r.register_forward_hook(_hook))
        for u in (getattr(block, "upsamplers", None) or []):
            handles.append(u.register_forward_hook(_hook))
        try:
            result = call()
        finally:
            for h in handles:
                h.remove()
        return result, outs

    def forward(self, sample, timestep, encoder_hidden_states, brushnet_cond,
                conditioning_scale=1.0, return_dict=False):
        # 1. time（官方同款：f32 sinusoid → cast sample dtype）
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], device=sample.device)
        elif timesteps.ndim == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        t_emb = self.time_proj(timesteps).to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, None)

        # 2. pre-process：cat([noisy, cond_latents, mask]) → conv_in_condition
        sample = self.conv_in_condition(torch.cat([sample, brushnet_cond], 1))

        # 3. down
        down_block_res_samples = (sample,)
        for block in self.down_blocks:
            if getattr(block, "has_cross_attention", False):
                sample, res_samples = block(
                    hidden_states=sample, temb=emb,
                    encoder_hidden_states=encoder_hidden_states)
            else:
                sample, res_samples = block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples
        down_out = tuple(
            b(r) for r, b in zip(down_block_res_samples,
                                 self.brushnet_down_blocks))

        # 4. mid
        if getattr(self.mid_block, "has_cross_attention", False):
            sample = self.mid_block(
                sample, emb, encoder_hidden_states=encoder_hidden_states)
        else:
            sample = self.mid_block(sample, emb)
        mid_out = self.brushnet_mid_block(sample)

        # 5. up（hook 捕获；官方 fork 是魔改 UpBlock return_res_samples）
        up_res_samples = ()
        for i, block in enumerate(self.up_blocks):
            is_final_block = i == len(self.up_blocks) - 1
            res_samples = down_block_res_samples[-len(block.resnets):]
            down_block_res_samples = \
                down_block_res_samples[:-len(block.resnets)]
            upsample_size = None if is_final_block \
                else down_block_res_samples[-1].shape[2:]
            if getattr(block, "has_cross_attention", False):
                def _call(b=block, s=sample, rs=res_samples, us=upsample_size):
                    return b(hidden_states=s, temb=emb,
                             res_hidden_states_tuple=rs,
                             encoder_hidden_states=encoder_hidden_states,
                             upsample_size=us)
            else:
                def _call(b=block, s=sample, rs=res_samples, us=upsample_size):
                    return b(hidden_states=s, temb=emb,
                             res_hidden_states_tuple=rs, upsample_size=us)
            sample, outs = self._capture_up_res(block, _call)
            up_res_samples += tuple(outs)
        up_out = tuple(
            b(r) for r, b in zip(up_res_samples, self.brushnet_up_blocks))

        # 6. scaling（guess_mode/global_pool 不支持也不需要——官方推理默认）
        down_out = [r * conditioning_scale for r in down_out]
        mid_out = mid_out * conditioning_scale
        up_out = [r * conditioning_scale for r in up_out]
        return down_out, mid_out, up_out


# ---------------------------------------------------------------------------
# 权重解析与加载
# ---------------------------------------------------------------------------

def _models_root():
    return os.environ.get("MODELS_DIR", "/models")


def _resolve_brushnet_path(name):
    """候选：MODELS_DIR/brushnet/<name>、MODELS_DIR/<name>、name 原样。"""
    root = _models_root()
    cands = [os.path.join(root, "brushnet", name),
             os.path.join(root, name),
             name]
    for p in cands:
        if os.path.exists(p):
            return p
    raise ValueError(f"brushnet 权重未找到：{name}（候选：{cands}）")


def _sniff_config_from_state(sd):
    """无 config.json 时按 state_dict 键嗅探结构（PowerPaint_Brushnet 裸权重）。"""
    n_down = 1 + max(int(k.split(".")[1]) for k in sd
                     if k.startswith("down_blocks."))
    block_out = [int(sd[f"down_blocks.{i}.resnets.0.norm1.weight"].shape[0])
                 for i in range(n_down)]
    lpb = len({int(k.split(".")[2]) for k in sd
               if k.startswith("down_blocks.0.resnets.")})

    def _attn(prefix):
        return any(k.startswith(prefix) and ".attentions." in k for k in sd)

    down_types = ["CrossAttnDownBlock2D" if _attn(f"down_blocks.{i}.")
                  else "DownBlock2D" for i in range(n_down)]
    up_types = ["CrossAttnUpBlock2D" if _attn(f"up_blocks.{i}.")
                else "UpBlock2D" for i in range(n_down)]
    mid_type = ("UNetMidBlock2DCrossAttn" if _attn("mid_block.")
                else "MidBlock2D")
    cross_dim = None
    for k in sd:
        if ".attentions.0.to_k.weight" in k:
            cross_dim = int(sd[k].shape[1])
            break
    cfg = {
        "in_channels": 4,
        "conditioning_channels": 5,
        "down_block_types": down_types,
        "mid_block_type": mid_type,
        "up_block_types": up_types,
        "block_out_channels": block_out,
        "layers_per_block": lpb,
        "cross_attention_dim": cross_dim or 768,
        "attention_head_dim": 8,
        "act_fn": "silu",
        "norm_num_groups": 32,
        "norm_eps": 1e-5,
        "brushnet_conditioning_channel_order": "rgb",
    }
    print(f"[brushnet] 无 config.json，按键嗅探：blocks={n_down} "
          f"channels={block_out} down={down_types} mid={mid_type} "
          f"cross_dim={cfg['cross_attention_dim']}", flush=True)
    return cfg


def _load_brushnet_files(path, torch_dtype):
    """返回 (config, state_dict, weights_bytes)。"""
    import json
    from safetensors.torch import load_file
    cfg = None
    sd_path = None
    if os.path.isdir(path):
        cfg_file = os.path.join(path, "config.json")
        if os.path.isfile(cfg_file):
            with open(cfg_file) as f:
                cfg = json.load(f)
        for cand in ("diffusion_pytorch_model.fp16.safetensors",
                     "diffusion_pytorch_model.safetensors"):
            p = os.path.join(path, cand)
            if os.path.isfile(p):
                sd_path = p
                break
        if sd_path is None:
            sts = [f for f in sorted(os.listdir(path))
                   if f.endswith(".safetensors")]
            if sts:
                sd_path = os.path.join(path, sts[0])
        if sd_path is None:
            raise ValueError(f"{path} 下无 safetensors 权重")
    else:
        sd_path = path
    sd = load_file(sd_path)
    if cfg is None:
        cfg = _sniff_config_from_state(sd)
    wbytes = os.path.getsize(sd_path)
    # config 里的列表/元组统一；丢弃与本实现无关的键
    keep = {"in_channels", "conditioning_channels", "flip_sin_to_cos",
            "freq_shift", "down_block_types", "mid_block_type",
            "up_block_types", "only_cross_attention", "block_out_channels",
            "layers_per_block", "downsample_padding", "mid_block_scale_factor",
            "act_fn", "norm_num_groups", "norm_eps", "cross_attention_dim",
            "transformer_layers_per_block", "attention_head_dim",
            "num_attention_heads", "use_linear_projection", "upcast_attention",
            "resnet_time_scale_shift", "brushnet_conditioning_channel_order"}
    cfg = {k: v for k, v in cfg.items() if k in keep}
    cfg.setdefault("in_channels", 4)
    cfg.setdefault("conditioning_channels", 5)
    return cfg, sd, wbytes


class _BrushNetBundle:
    """brushnet.apply 产物：brushnet 模型 + 预算好的 5ch 条件 + 强度/步窗口。

    cond 存 CPU fp32（采样时再搬执行设备/cast），与 ControlBundle hint 存
    PIL 的驻留哲学一致（apply 时不知采样设备）。
    """

    def __init__(self, brushnet, cond, scale, start_percent, end_percent):
        self.brushnet = brushnet
        self.cond = cond
        self.scale = float(scale)
        self.start_percent = float(start_percent)
        self.end_percent = float(end_percent)


# ---------------------------------------------------------------------------
# 主 UNet 块内残差注入（forward hook 精确复刻官方 fork 语义）
# ---------------------------------------------------------------------------

def _unwrap_out(out):
    """hook 输出解包：tensor / dataclass(.sample) / tuple 皆可。"""
    if torch.is_tensor(out):
        return out
    if hasattr(out, "sample") and torch.is_tensor(out.sample):
        return out.sample
    if isinstance(out, (tuple, list)) and out:
        return out[0]
    raise TypeError(f"无法解包模块输出: {type(out)}")


def _repack_out(out, new_t):
    if torch.is_tensor(out):
        return new_t
    if hasattr(out, "sample") and torch.is_tensor(out.sample):
        out.sample = new_t
        return out
    if isinstance(out, (tuple, list)) and out:
        return (new_t,) + tuple(out[1:])
    raise TypeError(f"无法重组模块输出: {type(out)}")


def _make_pre_add_hook(res, mod_name):
    """forward_pre_hook：改模块入参，等价「主路径加残差、skip 不加」。

    fork 语义：down_block_res_samples=(conv_in 输出,) 先存 skip，之后才
    sample += down[0] 进 down path——所以 conv_in 位不能改 conv_in 输出
    （会让 skip 通道也带残差），必须挂在 down_blocks[0].resnets[0] 入口改
    隐藏态（2026-11-19 分级差分实测 down[0] 单位发散 0.21 后修正）。
    """
    def hook(_m, args, kwargs):
        t = args[0]
        if not torch.is_tensor(t) or t.shape != res.shape:
            raise RuntimeError(
                f"brushnet 残差形状不符 @{mod_name}: 模块入参 "
                f"{tuple(t.shape) if torch.is_tensor(t) else type(t)} "
                f"vs 残差 {tuple(res.shape)}")
        return (t + res,) + tuple(args[1:]), kwargs
    return hook


def _make_add_hook(res, mod_name):
    def hook(_m, _a, out):
        t = _unwrap_out(out)
        if t.shape != res.shape:
            raise RuntimeError(
                f"brushnet 残差形状不符 @{mod_name}: 模块输出 "
                f"{tuple(t.shape)} vs 残差 {tuple(res.shape)}")
        return _repack_out(out, t + res)
    return hook


class _Injector:
    """把 brushnet 残差按官方位置挂到主 UNet 子模块的 forward 后钩子上。

    注入顺序（与 fork unet_2d_condition.py 的 pop(0) 顺序逐项对齐）。
    ⚠️ CrossAttn 块内 fork 语义是 resnet→attn→+res（残差加在 attention
    之后）——因此带 attention 的块 hook 挂 attentions[i]（在 resnet 之后
    立即执行），无 attention 的块才挂 resnet 本体。挂错层级会让注入信号
    先经 attn 非线性变换，位置不等价（2026-11-19 注入差分实测 maxdiff
    0.247 后修正）。
    """

    @staticmethod
    def _block_units(blk, is_up=False):
        """一个 block 内应挂 hook 的模块序列（与 fork 注入位一一对应）。"""
        atths = getattr(blk, "attentions", None)
        n_res = len(blk.resnets)
        units = []
        if atths and len(atths) == n_res:
            units.extend(atths)          # resnet→attn 成对，残差加 attn 后
        else:
            units.extend(blk.resnets)    # 无 attn：残差加 resnet 后
        units.extend(getattr(blk, "downsamplers", None) or [] if not is_up
                     else (getattr(blk, "upsamplers", None) or []))
        return units

    def __init__(self, unet, down_res, mid_res, up_res):
        plan = []
        it = iter(down_res)
        # down[0]（conv_in 位）：pre-hook 挂 down_blocks[0] 首个 resnet 入口
        # ——fork 是「先存 skip 再加残差」，改 conv_in 输出会污染 skip 通道
        first = unet.down_blocks[0].resnets[0]
        plan.append(("pre", first, next(it)))
        for blk in unet.down_blocks:
            for m in self._block_units(blk):
                plan.append((m, next(it)))
        try:
            extra = next(it)
            raise ValueError(
                f"brushnet down 残差比 UNet 结构多（如 {tuple(extra.shape)}）"
                "——权重与底模块结构不匹配")
        except StopIteration:
            pass
        plan.append((unet.mid_block, mid_res))
        it = iter(up_res)
        for blk in unet.up_blocks:
            for m in self._block_units(blk, is_up=True):
                plan.append((m, next(it)))
        try:
            extra = next(it)
            raise ValueError(
                f"brushnet up 残差比 UNet 结构多（如 {tuple(extra.shape)}）"
                "——权重与底模块结构不匹配")
        except StopIteration:
            pass
        self._plan = plan
        self._handles = []

    def __enter__(self):
        for item in self._plan:
            if item[0] == "pre":
                _, m, res = item
                self._handles.append(m.register_forward_pre_hook(
                    _make_pre_add_hook(res, type(m).__name__),
                    with_kwargs=True))
            else:
                m, res = item
                self._handles.append(m.register_forward_hook(
                    _make_add_hook(res, type(m).__name__)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


# ---------------------------------------------------------------------------
# PowerPaint task prompt
# ---------------------------------------------------------------------------

# (占位符顺序 = 权重行序，与官方 add_tokens(["P_ctxt","P_shape","P_obj"]) 一致)
_PLACEHOLDERS = ("P_ctxt", "P_shape", "P_obj")
_VECS_PER_TOKEN = 10
# task → (positive 后缀, negative 后缀)，映射表照抄官方 app.py add_task
# （A/B 两变体取 A 侧；outpainting 与 object-removal 官方同配方）
_TASK_SUFFIX = {
    "object_removal": ("P_ctxt", "P_obj"),
    "outpainting": ("P_ctxt", "P_obj"),
    "object_insertion": ("P_obj", "P_obj"),
    "shape_guided": ("P_shape", "P_shape"),
}


def _task_prompt_path():
    return os.path.join(_models_root(), "task_prompts",
                        "powerpaint.safetensors")


def _ensure_task_tokens(pipe):
    """把 30 个 task token + 学习到的 embedding 注入 tokenizer/text_encoder
    （幂等：LRU 常驻管道只注一次；额外 token 不影响普通提示词编码）。"""
    if getattr(pipe, "_flowx_task_tokens", False):
        return
    from safetensors.torch import load_file
    path = _task_prompt_path()
    if not os.path.isfile(path):
        raise ValueError(
            f"task prompt 权重缺失：{path}（应从 PowerPaint 微调 "
            f"text_encoder 尾部 30 行抽取，见族文件 docstring）")
    emb = load_file(path)["embeddings"]  # [30, 768]
    names = [f"{p}_{i}" for p in _PLACEHOLDERS for i in range(_VECS_PER_TOKEN)]
    if emb.shape[0] != len(names):
        raise ValueError(f"{path} embeddings 行数 {emb.shape[0]} ≠ {len(names)}")
    tok, te = pipe.tokenizer, pipe.text_encoder
    tok.add_tokens(names)
    te.resize_token_embeddings(len(tok))
    with torch.no_grad():
        w = te.get_input_embeddings().weight
        w[-emb.shape[0]:] = emb.to(device=w.device, dtype=w.dtype)
    pipe._flowx_task_tokens = True
    print(f"[clip.encode-task] task tokens 注入完成（+{len(names)} tokens，"
          f"embedding 来自 {os.path.basename(path)}）", flush=True)


# ---------------------------------------------------------------------------
# 算子注册
# ---------------------------------------------------------------------------

def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "sd.brushnet.load",
        inputs={"name": "STRING", "dtype": "STRING"},
        outputs={"control_net": "CONTROL_NET"},
        description="BrushNet 加载：MODELS_DIR/brushnet/<name>（diffusers 组件目录"
                    " 或 safetensors 单文件；无 config.json 时按键嗅探结构，兼容 "
                    "PowerPaint_Brushnet 裸权重）。dtype=auto 默认 fp16，进模型 "
                    "LRU 常驻。输出经 brushnet.apply 捆条件后喂 sample_brushnet")
    def _op_brushnet_load(name, dtype="auto"):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化
        dt = "fp16" if dtype in (None, "", "auto") else str(dtype)
        key = f"brushnet:{name}|dtype={dt}"
        cached = models._pipes.get(key)
        if cached is not None:
            print(f"[brushnet.load] LRU 命中 {key}", flush=True)
            return {"control_net": cached}
        torch_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16,
                       "fp32": torch.float32}[dt]
        path = _resolve_brushnet_path(name)
        t0 = time.time()
        cfg, sd, wbytes = _load_brushnet_files(path, torch_dtype)
        model = BrushNetModel(cfg)
        model.to(dtype=torch_dtype)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        del sd
        print(f"[brushnet.load] {name}: missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
        if missing:
            raise ValueError(
                f"brushnet 权重缺 {len(missing)} 键（结构推断与实际权重不符），"
                f"样例：{sorted(missing)[:5]}")
        if unexpected:
            print(f"[brushnet.load] ⚠️ {len(unexpected)} 个权重键未使用，"
                  f"样例：{sorted(unexpected)[:5]}", flush=True)
        model.eval()
        models.register_external(key, model, weights_bytes=wbytes,
                                 arch="BrushNetModel", offload_mode="none")
        print(f"[brushnet.load] {name} 加载完成（{time.time()-t0:.1f}s，"
              f"dtype={dt}）", flush=True)
        return {"control_net": model}

    @registry.register(
        "sd.brushnet.apply",
        inputs={"control_net": "CONTROL_NET", "vae": "VAE", "image": "IMAGE",
                "mask": "IMAGE", "scale": "FLOAT", "start_percent": "FLOAT",
                "end_percent": "FLOAT"},
        outputs={"control": "CONTROL"},
        description="BrushNet 捆条件：官方语义 masked_image=image×(1-mask)（mask "
                    "白=重绘区）→ VAE 编码 4ch + mask 通道（1=保留区，即 1-mask）"
                    "插值到 latent 尺寸 1ch = "
                    "5ch 条件。scale=残差强度（1.0 官方默认），start/end_percent "
                    "为生效步窗口。产物喂 sample_brushnet 的 control 端口")
    def _op_brushnet_apply(control_net, vae, image, mask, scale=1.0,
                           start_percent=0.0, end_percent=1.0):
        import numpy as np
        pipe, _ = ops.resolve_pipe(vae)
        img = image.convert("RGB")
        w, h = img.size
        msk = mask.convert("L")
        if msk.size != (w, h):
            msk = msk.resize((w, h))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        m = (np.asarray(msk, dtype=np.float32) / 255.0 > 0.5
             ).astype(np.float32)[..., None]
        # 官方：masked_image = original × (1-mask)（重绘区置黑）
        masked = arr * (1.0 - m)
        x = torch.from_numpy(masked).permute(2, 0, 1)[None] * 2.0 - 1.0
        device = ops.exec_device_of(pipe)
        x = x.to(device=device, dtype=pipe.vae.dtype)
        pipe.vae.enable_tiling()
        try:
            with torch.no_grad():
                lat = (pipe.vae.encode(x).latent_dist.mean
                       * pipe.vae.config.scaling_factor)
        finally:
            pipe.vae.disable_tiling()
        mt = torch.from_numpy(m[:, :, 0])[None, None].to(device=lat.device)
        # 官方：interpolate 到 latent 尺寸（默认最近邻）
        mt = F.interpolate(mt, size=tuple(lat.shape[-2:]), mode="nearest")
        # ⚠️ 官方 cond mask 通道语义：1=保留区、0=重绘区（fork pipeline 将
        # mask 归一化 [-1,1] 后 sum<0 判定：黑=1）。本节点 mask 白=重绘，
        # 故通道取 1-mt（2026-11-19 逐步 spy 差分 STEP0-cond maxdiff=1.0 后修正）
        mt = 1.0 - mt
        cond = torch.cat([lat.float(), mt.float()], dim=1).cpu()
        print(f"[brushnet.apply] image={w}x{h} scale={scale} "
              f"window=[{start_percent},{end_percent}) "
              f"cond={tuple(cond.shape)} mask占比={m.mean().item():.3f}",
              flush=True)
        return {"control": _BrushNetBundle(control_net, cond, scale,
                                           start_percent, end_percent)}

    @registry.register(
        "sd.sample_brushnet",
        inputs={"model": "MODEL", "pos": "COND", "neg": "COND",
                "latent": "LATENT", "control": "CONTROL",
                "brushnet_pos": "COND", "brushnet_neg": "COND",
                "seed": "INT", "steps": "INT", "cfg": "FLOAT",
                "sampler_name": "STRING", "scheduler": "STRING",
                "denoise": "FLOAT", "start_at_step": "INT",
                "end_at_step": "INT", "add_noise": "BOOL",
                "cn_control": "CONTROL", "preview_every": "INT"},
        outputs={"latent": "LATENT", "seed": "INT"},
        description="BrushNet KSampler：采样循环内每步先跑 brushnet 分支，残差经 "
                    "hook 块内注入主 UNet（官方 fork 语义），其余能力（分段采样/"
                    "noise_mask/ControlNet 叠加/预览）与 sd.sample 一致。"
                    "brushnet_pos/neg（可选）：brushnet 分支专用 cond"
                    "（PowerPaint task prompt），缺省与主 cond 相同；"
                    "cn_control（可选）：常规 ControlNet 链，与 brushnet 正交叠加")
    def _op_sample_brushnet(model, pos, neg, latent, control=None,
                            brushnet_pos=None, brushnet_neg=None,
                            seed=-1, steps=20, cfg=7.0,
                            sampler_name="euler", scheduler="normal",
                            denoise=1.0, start_at_step=0, end_at_step=0,
                            add_noise=True, cn_control=None, preview_every=1):
        if control is None or not hasattr(control, "brushnet"):
            raise ValueError(
                "sd.sample_brushnet: control 必须是 brushnet.apply 的产物"
                "（常规 ControlNet 请接 cn_control）")
        pipe, _patches = ops.resolve_pipe(model)
        unet = pipe.unet
        device = ops.exec_device_of(pipe)
        bn = control.brushnet
        cond = control.cond.to(device=device, dtype=unet.dtype)

        base_samples, _nm = ops.resolve_latent(latent)
        b = base_samples.shape[0]
        if cond.shape[0] not in (b, 1):
            raise ValueError(
                f"brushnet 条件 batch {cond.shape[0]} 与 latent batch {b} 不符")

        # 步窗口（复刻框架 resolve_span 语义，窗口按实际执行步数计）
        start, end = ops.resolve_span(steps, denoise, start_at_step, end_at_step)
        total = end - start
        win_lo = round(control.start_percent * total)
        win_hi = round(control.end_percent * total)

        # 前向模式推断（决定每步 unet 调用次数与 hidden 选边）
        pos_segs = ops.as_cond_segments(pos, "pos")
        neg_segs = ops.as_cond_segments(neg, "neg")
        plain = (len(pos_segs) == 1 and len(neg_segs) == 1
                 and pos_segs[0][1] is None and neg_segs[0][1] is None)
        if not plain:
            raise ValueError(
                "brushnet 暂不与 cond.set_area 组合（区域条件需逐段跑 brushnet，"
                "成本×N）；请仅用整图 cond")
        eq_len = pos_segs[0][0].shape[1] == neg_segs[0][0].shape[1]
        cpp = 1 if eq_len else 2  # cat=1 次/步；split=2 次/步（先 neg 后 pos）

        # brushnet 分支专用 cond（PowerPaint task prompt）；缺省跟随主 cond
        bn_pos_e = bn_neg_e = None
        if brushnet_pos is not None or brushnet_neg is not None:
            segp = ops.as_cond_segments(
                brushnet_pos if brushnet_pos is not None else pos,
                "brushnet_pos")
            segn = ops.as_cond_segments(
                brushnet_neg if brushnet_neg is not None else neg,
                "brushnet_neg")
            if (len(segp) != 1 or len(segn) != 1
                    or segp[0][1] is not None or segn[0][1] is not None):
                raise ValueError("brushnet_pos/neg 需为整图 cond")
            bn_pos_e = segp[0][0]
            bn_neg_e = segn[0][0]
            print(f"[sample-brushnet] brushnet 分支用专用 cond "
                  f"(pos={tuple(bn_pos_e.shape)} neg={tuple(bn_neg_e.shape)})",
                  flush=True)

        def _bn_hidden(sample, fallback, call_idx):
            if bn_pos_e is None:
                return fallback
            nb = sample.shape[0]
            if nb == 2 * b:
                return torch.cat([
                    bn_neg_e.to(device=sample.device, dtype=sample.dtype),
                    bn_pos_e.to(device=sample.device, dtype=sample.dtype)])
            # 单侧调用：split 模式框架先 neg 后 pos（ops.py blend_side/unet_forward
            # 顺序实证）；cat 模式 cfg<=1 时只有 pos
            e = bn_neg_e if (cpp == 2 and call_idx % 2 == 0) else bn_pos_e
            return e.to(device=sample.device, dtype=sample.dtype)

        hint = "sd15"
        if preview_every > 0:
            hint = preview.model_hint_of(pipe)

        def on_step(latents, i, tot):
            job = execution.current()
            if job is None:
                return
            job.set_progress(i + 1, tot)
            if preview_every > 0:
                rec = preview.recorder_for(job.id, preview_every, hint)
                if rec.want(i, tot):
                    rec.push(latents, (i + 1) / tot)

        orig_forward = unet.forward
        calls = {"n": 0}

        def patched(sample, timestep, encoder_hidden_states=None, **kwargs):
            n = calls["n"]
            calls["n"] = n + 1
            i = n // cpp
            if not (win_lo <= i < win_hi):
                return orig_forward(
                    sample, timestep,
                    encoder_hidden_states=encoder_hidden_states, **kwargs)
            hidden = _bn_hidden(sample, encoder_hidden_states, n)
            bc = cond if cond.shape[0] == sample.shape[0] \
                else cond.expand(sample.shape[0], -1, -1, -1)
            down, mid, up = bn(sample, timestep, hidden, bc,
                               conditioning_scale=control.scale,
                               return_dict=False)
            with _Injector(unet, down, mid, up):
                return orig_forward(
                    sample, timestep,
                    encoder_hidden_states=encoder_hidden_states, **kwargs)

        # BrushNet 驻留：采样期间搬上执行设备，结束搬回（SD1.5 fp16 ~1.5GB，
        # 8GB 卡可承受；比逐调用搬移省 N 次传输）
        prev_dev = next(bn.parameters()).device
        bn.to(device)
        unet.forward = patched
        print(f"[sample-brushnet] steps={steps} window=[{win_lo},{win_hi})/{total} "
              f"scale={control.scale} cpp={cpp} device={device}", flush=True)
        try:
            return ops.sample(
                model, pos, neg, latent, seed, steps, cfg, sampler_name,
                scheduler, denoise, start_at_step, end_at_step, add_noise,
                control=cn_control, preview_cb=on_step,
                interrupt_check=execution.check_cancelled)
        finally:
            unet.forward = orig_forward
            try:
                bn.to(prev_dev)
            except Exception:
                pass
            if device.type == "cuda":
                torch.cuda.empty_cache()

    @registry.register(
        "sd.clip.encode_task",
        inputs={"clip": "CLIP", "text": "STRING", "task": "STRING",
                "side": "STRING", "clip_skip": "INT"},
        outputs={"cond": "COND"},
        description="CLIP Text Encode + PowerPaint task prompt：task 可选 "
                    "none/object_removal/object_insertion/outpainting/"
                    "shape_guided，side=positive/negative；自动注入 P_ctxt/"
                    "P_shape/P_obj task token（embedding 来自 MODELS_DIR/"
                    "task_prompts/powerpaint.safetensors），其余语义同 "
                    "sd.clip.encode。task=none 时与原版完全一致")
    def _op_clip_encode_task(clip, text, task="none", side="positive",
                             clip_skip=0):
        pipe, _ = ops.resolve_pipe(clip)
        if task and str(task) != "none":
            if task not in _TASK_SUFFIX:
                raise ValueError(
                    f"未知 task '{task}'（可选：none/"
                    f"{'/'.join(_TASK_SUFFIX)}）")
            suffix = _TASK_SUFFIX[task][0 if side == "positive" else 1]
            _ensure_task_tokens(pipe)
            toks = " ".join(f"{suffix}_{i}" for i in range(_VECS_PER_TOKEN))
            text = f"{text} {toks}"
            print(f"[clip.encode-task] task={task} side={side} → "
                  f"后缀 {suffix}×{_VECS_PER_TOKEN}", flush=True)
        kw = {"clip_skip": clip_skip} if clip_skip else {}
        return ops.clip_encode(pipe, text, **kw)
