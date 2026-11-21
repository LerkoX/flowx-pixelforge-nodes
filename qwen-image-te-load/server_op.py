"""Qwen-Image 算子族：阿里通义千问图像生成（T2I，20B MMDiT）。

随节点包自注册（与 wan/minimax_h3 同款模式）。

模型事实（对拍 diffusers 0.42.1 pipeline_qwenimage.py）：
- 组件五件：QwenImageTransformer2DModel（20B，60 层，in_channels=64）
  + Qwen2_5_VLForConditionalGeneration（7B 文本编码器，**必须 bf16**——
  T5/Qwen 系 fp16 会 LayerNorm 溢出，Wan 真机教训）
  + AutoencoderKLQwenImage（VAE ~243MB，**官方示例 fp32**）
  + Qwen2Tokenizer + FlowMatchEulerDiscreteScheduler
- CFG 形态：guidance_embeds=false → guidance_scale 无效；负提示走
  true_cfg_scale（官方默认 4.0，>1 且给了 negative_prompt 才生效）
- prompt 模板（<|im_start|>system…Describe the image…）由 pipeline 内部
  处理，算子透传 prompt 即可；max_sequence_length ≤ 1024
- 分辨率：VAE 8 倍 × patch 2 = 边长须 16 的倍数（官方推荐 1328×1328、
  1664×928、928×1664、1472×1140、1140×1472、1296×1728(2512) 等）
- Lightning 加速 LoRA（lightx2v/Qwen-Image-Lightning，diffusers 格式）：
  4/8 步出图，挂 LoRA 后 true_cfg_scale 用 1.0（蒸馏模型不吃 CFG）

权重布局（QWEN_IMAGE_DIR，默认 /models/qwen-image；优先共享盘软链）：
- <model>/{transformer,text_encoder,tokenizer,vae,scheduler}/
    官方 Diffusers 目录格式（ModelScope/HF Qwen/Qwen-Image[-2512] 整仓库软链）
- diffusion_models/<file>.safetensors   Comfy-Org 单文件（bf16/fp8_e4m3fn，
    走 from_single_file 尽力转换；fp8 带 scale_weight 的 Comfy 布局若转换
    失败会明确报错并建议 bf16 目录路线）
- loras/<file>.safetensors               Lightning 等 diffusers 格式 LoRA

显存/内存相位（24GB 卡 + 64GB 内存）：
- transformer 20B bf16 ≈ 39GB > 24GB ⇒ 逐层 offload（comfy_quant 的
  _sequential_offload，CPU 常驻，峰值仅单层+激活；Wan 14B 同款已实证）
- TE 7B bf16 ≈ 16.5GB ⇒ 同样逐层 offload（编码完可留 CPU）
- VAE fp32 ≈ 0.5GB ⇒ 常驻上卡
- CPU 常驻合计 ≈ 56GB << 64GB（H3 是 58GB 跑通过）；跑前若 H3/Wan 组件
  常驻须先卸载腾内存

算子（拆分式，与 wan 同粒度）：
- qwen_image.transformer_load / te_load / vae_load → 组件对象
- qwen_image.pipe(transformer, text_encoder, vae, model, lora, lora_scale,
  max_sequence_length) → MODEL（QwenImagePipeline + LoRA）
- qwen_image.gen(model, prompt, negative_prompt, steps, true_cfg_scale,
  width, height, seed, max_sequence_length, num_images) → IMAGE + seed
"""

import os
import time

# --- 以下为 comfy_quant.py 内联（sync-common 自动生成，勿改）---
"""comfy_quant.py：ComfyUI 量化权重格式（int8_convrot 等）通用适配层。

从 minimax_h3 实战抽取（2026-11-08 真机出片验证）：Comfy-Org 发布的
int8_convrot safetensors 权重包 + diffusers/transformers 官方组件之间的
通用翻译件。后续遇到同格式模型（Wan/Hunyuan/其他 Comfy-Org int8 包）
直接复用，模型侧只需写 key 映射 + 装配逻辑。

内容：
- ConvRot 旋转量化：_hadamard / _rotate_weight_convrot（与 comfy_kitchen
  dequantize_int8_convrot_weight 逐位等价——量化器 W_rot=W@Hᵀ 后 int8，
  dequant 侧必须逆旋转回原始基，否则全模型输出乱码。H3 真机花屏根因）
- _Int8DequantLinear：int8 常驻 + per-row scale + convrot 逆旋转 +
  LoRA 旁路的 Linear 替换件（内存 1/4，与 bf16 全量推理逐层 dequant 等价）
- _split_quant_sd：读 int8_convrot safetensors → (base, scale, convrot)
  三元组（.comfy_quant metadata U8→JSON 解析，groupsize 按层提取——
  注意不全是 256，H3 的 adaln 层是 64）
- _rename_scales_like：scale/convrot dict 按 weight 同规则改名
- _materialize_mixed：meta 模型逐参数物化（int8 替换 + assign=True 灌入 +
  rope inv_freq 公式重算 + zero_fill 兜底）——全程无全量 bf16 实例化
- _dequant_safetensors / _load_dequant_and_materialize：全量反量化直载路径
  （小体积组件用，如 VAE 回退）
- _sequential_offload：逐层 CPU offload（自写 forward hook，post 零拷贝——
  accelerate 整模型 offload 在超大 TE 场景 OOM 且 encoders 直调子模块会
  绕过 post_forward，真机实测）

⚠️ 分发机制：本文件不是算子族（无 register()），不进 OP_FAMILIES；族文件
（如 minimax_h3.py）写 `from comfy_quant import ...`，sync-common.py 分发时
把本文件内容内联进单文件 server_op.py（插件上传是单文件机制）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

_DEQUANT_DTYPE = "bfloat16"


# ---------------------------------------------------------------------------
# ConvRot（Hadamard 旋转量化）
# ---------------------------------------------------------------------------

_HADAMARD_CACHE = {}


def _hadamard(size):
    """ConvRot 的正规正交 Hadamard 矩阵（与 comfy_kitchen 逐位一致）：
    h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]，kron 迭代至 size，
    /sqrt(size) 归一化。对称且对合（H@H=I）——量化/逆旋转共用同一操作。
    参考 comfy_kitchen/tensor/int8_utils.py:_build_hadamard。"""
    import math
    cached = _HADAMARD_CACHE.get(size)
    if cached is not None:
        return cached
    if size < 4 or (size & (size - 1)) != 0 or math.log(size, 4) % 1 != 0:
        raise ValueError(f"ConvRot Hadamard size 必须是 4 的幂，got {size}")
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1],
                       [-1, 1, 1, 1]], dtype=torch.float32)
    h = h4
    cur = 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[size] = h
    return h


def _rotate_weight_convrot(w, group_size):
    """ConvRot 权重旋转（w: (out, in) fp32）：按输入维分组右乘 H^T。

    量化器：W_rot = W @ Hᵀ 后 int8 量化；dequant 侧用同一操作转回原始基
    （H 对称对合）。参考 comfy_kitchen int8_utils._rotate_weight /
    dequantize_int8_convrot_weight。"""
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(
            f"convrot: in_features {in_f} 不整除 group_size {group_size}")
    h = _hadamard(group_size).to(device=w.device)
    return torch.matmul(
        w.reshape(out_f, in_f // group_size, group_size),
        h.T).reshape(out_f, in_f)


class _Int8DequantLinear(nn.Module):
    """int8 weight + per-output-channel fp32 scale 的 Linear 替换件。

    权重以 int8 buffer 常驻（1/4 体积，随 offload hook 上下卡）；forward 时
    dequant 到激活 dtype 后 F.linear——与 bf16 全量推理逐位等价（dequant 是
    确定映射）。bias 走非量化路径（bf16/fp32 直载）。

    convrot_groupsize>0 时：dequant 后先做 ConvRot 逆旋转（按输入维分组
    Hadamard）再线性——这是 int8_convrot checkpoint 的正确语义，不转回
    原始基则全模型输出乱码（H3 真机实测：VAE 靠 scale≈0 残差直通掩盖，
    transformer 全量化层 → 纯噪声花屏）。旋转在 fp32 域做再 cast，
    与 comfy_kitchen dequantize_int8_convrot_weight 等价。"""

    def __init__(self, weight_int8, weight_scale, bias=None,
                 convrot_groupsize=0):
        super().__init__()
        if weight_int8.dtype != torch.int8:
            raise TypeError(f"weight_int8 需 int8，got {weight_int8.dtype}")
        self.register_buffer("weight_int8", weight_int8.contiguous())
        self.register_buffer("weight_scale",
                             weight_scale.float().reshape(-1, 1).contiguous())
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.out_features, self.in_features = weight_int8.shape
        self.convrot_groupsize = int(convrot_groupsize)
        if self.convrot_groupsize > 0 and \
                self.in_features % self.convrot_groupsize != 0:
            # 不整除的层量化器也无法旋转——视为未旋转（防御）
            print(f"[comfy-quant] convrot 跳过：in={self.in_features} 不整除 "
                  f"{self.convrot_groupsize}", flush=True)
            self.convrot_groupsize = 0

    def add_lora(self, A, B, scale):
        """挂 LoRA 旁路（register_buffer：随 offload hook 一起上下卡）。

        A (rank,in) / B (out,rank)，前向追加 y += (x @ Aᵀ @ Bᵀ) × scale。
        """
        i = getattr(self, "_n_lora", 0)
        self.register_buffer(f"lora_A{i}", A.to(torch.bfloat16).contiguous())
        self.register_buffer(f"lora_B{i}", B.to(torch.bfloat16).contiguous())
        scales = getattr(self, "_lora_scales", [])
        scales.append(float(scale))
        self._lora_scales = scales
        self._n_lora = i + 1

    def forward(self, x):
        if self.convrot_groupsize > 0:
            w = self.weight_int8.float() * self.weight_scale
            w = _rotate_weight_convrot(w, self.convrot_groupsize)
            w = w.to(x.dtype)
        else:
            w = self.weight_int8.to(x.dtype) * self.weight_scale.to(x.dtype)
        y = F.linear(x, w, self.bias)
        for i, s in enumerate(getattr(self, "_lora_scales", [])):
            A = getattr(self, f"lora_A{i}")
            B = getattr(self, f"lora_B{i}")
            y = y + (x.to(A.dtype) @ A.t() @ B.t()).to(y.dtype) * s
        return y

    def extra_repr(self):
        return (f"in_features={self.in_features}, "
                f"out_features={self.out_features}, int8-dequant, "
                f"convrot={self.convrot_groupsize}")


# ---------------------------------------------------------------------------
# 量化 checkpoint 读取
# ---------------------------------------------------------------------------

def _split_quant_sd(path):
    """读 int8_convrot safetensors → (base_sd, scale_sd, convrot_sd)。

    base_sd：所有非 _scale/.comfy_quant key——量化层 weight 保持 int8 原样
    （不反量化！），非量化层（embedding/norm/bias/fp32 pinned）保留原 dtype。
    scale_sd：xxx.weight -> fp32 scale（per-row）。
    convrot_sd：xxx.weight -> hadamard group_size（0 = 未旋转）——从该层的
    .comfy_quant metadata（U8 字节序列 → JSON：{"format": "int8_tensorwise",
    "convrot": true, "convrot_groupsize": 256}）解析。ConvRot 旋转必须在
    dequant 后逆转回原始基，否则输出全乱（H3 真机花屏根因）。
    """
    import json as _json
    from safetensors import safe_open

    base, scales, convrot = {}, {}, {}
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for k in sorted(keys):
            if k.endswith(".comfy_quant"):
                try:
                    meta = _json.loads(
                        bytes(f.get_tensor(k).tolist()).decode("utf-8"))
                except Exception:
                    continue
                wkey = k[:-len(".comfy_quant")] + ".weight"
                if meta.get("convrot"):
                    convrot[wkey] = int(meta.get("convrot_groupsize", 256))
                continue
            if k.endswith("_scale"):
                scales[k[:-len("_scale")]] = f.get_tensor(k)
                continue
            base[k] = f.get_tensor(k)
    # 只保留确有 int8 配对的 scale（防孤儿 scale 误判替换）
    scales = {k: s for k, s in scales.items()
              if k in base and base[k].dtype == torch.int8}
    convrot = {k: g for k, g in convrot.items() if k in scales}
    return base, scales, convrot


def _rename_scales_like(scales, rename_pairs, special_handlers=None):
    """scale 字典按与 weight 相同的规则改名/变形（与官方转换函数平行）。

    rename_pairs：[(old_sub, new_sub)] 顺序子串替换（同官方 RENAME_DICT）；
    special_handlers：{匹配子串: fn(key, scales)}——如 QKV 拆分、gated FF 交换。"""
    out = dict(scales)
    for key in list(out.keys()):
        nk = key
        for old, new in rename_pairs:
            nk = nk.replace(old, new)
        if nk != key:
            out[nk] = out.pop(key)
    if special_handlers:
        for key in list(out.keys()):
            for pat, fn in special_handlers.items():
                if pat in key:
                    fn(key, out)
                    break
    return out


# ---------------------------------------------------------------------------
# meta 物化 / 直载
# ---------------------------------------------------------------------------

def _materialize_mixed(model, base_sd, scale_sd, name="",
                       zero_fill_missing=False, convrot_sd=None):
    """meta 模型逐参数物化：有 scale 的 nn.Linear → _Int8DequantLinear 替换
    （int8 常驻）；无 scale 参数 → 真实 tensor 灌入（meta .data 赋值）。

    全程无全量 bf16 实例化——峰值内存 ≈ int8 权重 + 非量化层，避免 63GB 爆内存。
    convrot_sd：{weight_key: group_size}，>0 的层 dequant 后需 Hadamard 逆旋转。
    返回 (替换层数, 仍缺权重的参数列表)。"""
    convrot_sd = convrot_sd or {}
    n_replaced = 0
    n_convrot = 0
    consumed = set()
    # ① 量化层替换
    for mod_name, mod in list(model.named_modules()):
        if not isinstance(mod, nn.Linear):
            continue
        wkey = f"{mod_name}.weight"
        if wkey not in scale_sd or wkey not in base_sd:
            continue
        w, s = base_sd[wkey], scale_sd[wkey]
        if w.shape != (mod.out_features, mod.in_features):
            raise ValueError(
                f"[{name}] {wkey} 形状不符：文件 {tuple(w.shape)} vs "
                f"模型 ({mod.out_features}, {mod.in_features})")
        bias = base_sd.get(f"{mod_name}.bias")
        gs = convrot_sd.get(wkey, 0)
        new = _Int8DequantLinear(w, s, bias, convrot_groupsize=gs)
        if gs > 0 and new.convrot_groupsize > 0:
            n_convrot += 1
        if "." in mod_name:
            parent_path, _, attr = mod_name.rpartition(".")
            parent = model.get_submodule(parent_path)
        else:
            attr = mod_name
            parent = model
        setattr(parent, attr, new)
        consumed.update((wkey, f"{mod_name}.bias"))
        n_replaced += 1
    if n_convrot:
        print(f"[quant-loader] {name}: {n_convrot}/{n_replaced} 层带 convrot "
              f"旋转（dequant 后 Hadamard 逆旋转）", flush=True)
    # ② 非量化参数物化：assign=True 直接把 meta 参数替换为 sd 的真实 tensor
    # （PyTorch 为 meta 模型加载设计的官方路径；.data= 赋实 tensor 会报
    # set_data 类型不兼容，真机实测）。
    rest = {k: v for k, v in base_sd.items() if k not in consumed}
    res = model.load_state_dict(rest, strict=False, assign=True)
    consumed.update(rest.keys())
    # ③ 计算型 buffer 物化（meta 实例化的后遗症）：rope 的 inv_freq 不随
    # checkpoint（从 config 重算，官方转换函数也直接丢弃它），meta 实例化后
    # 是空壳——.to() 即 'Cannot copy out of meta tensor'。
    # 按公式重算：inv_freq = 1 / theta^(arange(0, dim, 2) / dim)。
    # 【H3 真机花屏根因】diffusers 的 MiniMaxH3RotaryPosEmbed 官方公式除数是
    # 2*rope_freq_dim（arange(0, 2*rfd, 2)/(2*rfd)，16 个频率）——曾误用
    # rope_freq_dim（16）作除数 → 频率数减半且值全错 → patch 位置错位马赛克。
    # 约定：模块有 rope_freq_dim 属性时 dim=2*rfd（H3 系）；否则按标准
    # transformers 形式 buf.shape[0]*2（Qwen 系）。
    import torch as _t
    n_buf = 0
    for buf_name, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        parent_path, _, attr = buf_name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        if attr == "inv_freq":
            mod = parent
            rfd = getattr(mod, "rope_freq_dim", None)
            dim = 2 * rfd if rfd is not None else buf.shape[0] * 2
            theta = float(getattr(mod, "rope_theta",
                                  getattr(mod, "theta", 10000.0)))
            inv = 1.0 / (theta ** (_t.arange(0, dim, 2).float() / dim))
            parent._buffers[attr] = inv.to(buf.dtype if buf.dtype.is_floating_point
                                           else _t.float32)
        else:
            # 其他 meta buffer 兜底物化为零（非计算路径依赖项）
            parent._buffers[attr] = _t.zeros_like(buf, device="cpu")
        n_buf += 1
    if n_buf:
        print(f"[quant-loader] {name}: {n_buf} 个计算型 buffer 已重算物化",
              flush=True)
    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if missing and zero_fill_missing:
        # checkpoint 不含且推理不用的参数（lm_head / 被裁的 final norm）：
        # zeros 物化——不物化则 offload hook .to() 时 'Cannot copy out of meta
        # tensor' 爆炸（真机实测）。
        for pname in missing:
            parent_path, _, attr = pname.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            old = parent._parameters[attr]
            parent._parameters[attr] = torch.nn.Parameter(
                torch.zeros(old.shape, dtype=old.dtype), requires_grad=False)
        print(f"[quant-loader] {name}: {len(missing)} 个未用参数 zeros 物化",
              flush=True)
        missing = []
    leftover = [k for k in base_sd if k not in consumed
                and not k.endswith(".comfy_quant")]
    print(f"[quant-loader] {name}: int8 替换 {n_replaced} 层, "
          f"直载 {len(consumed) - n_replaced} 项, "
          f"缺权重 {len(missing)}, 未消费 {len(leftover)}", flush=True)
    if missing[:5]:
        print(f"  missing 样例: {missing[:5]}", flush=True)
    if leftover[:5]:
        print(f"  leftover 样例: {leftover[:5]}", flush=True)
    return n_replaced, missing


def _dequant_safetensors(path, dtype=None):
    """读 int8_convrot safetensors → 反量化 → 官方命名 state_dict（惰性逐 key）。

    量化层：xxx.weight(int8) + xxx.weight_scale(fp32, per-row) → W.float()*scale → bf16。
    非量化 key（embedding/norm/bias/fp32 层）原样保留（含原 dtype——load_state_dict
    的 copy_ 会按模型侧 dtype 自动 cast）。comfy_quant（ComfyUI 元数据）丢弃。
    ⚠️ 本函数不做 convrot 逆旋转——仅适用于未旋转量化层或小体积组件回退路径；
    带 convrot 的大模型请走 _split_quant_sd + _materialize_mixed。
    """
    from safetensors import safe_open

    out = {}
    target = dtype or getattr(torch, _DEQUANT_DTYPE)
    with safe_open(path, framework="pt") as f:
        keys = set(f.keys())
        for k in sorted(keys):
            if k.endswith(".comfy_quant") or k.endswith("_scale"):
                continue
            t = f.get_tensor(k)
            scale_key = k + "_scale"
            if t.dtype == torch.int8 and scale_key in keys:
                scale = f.get_tensor(scale_key)
                t = (t.float() * scale).to(target)
            out[k] = t
    return out


def _load_dequant_and_materialize(model, sd, strict=True):
    """把反量化后的 sd 灌进模型。模型在 CPU（from_config 实例化）；
    load_state_dict 的 copy_ 语义自动按参数 dtype cast（fp32 pinned 层由此保证）。
    返回 (missing, unexpected) 供诊断。"""
    res = model.load_state_dict(sd, strict=False)
    missing, unexpected = list(res.missing_keys), list(res.unexpected_keys)
    if missing or unexpected:
        print(f"[quant-loader] load_state_dict: missing={len(missing)} "
              f"unexpected={len(unexpected)}", flush=True)
        for k in missing[:5]:
            print(f"  missing: {k}", flush=True)
        for k in unexpected[:5]:
            print(f"  unexpected: {k}", flush=True)
    if strict and (missing or unexpected):
        raise ValueError(
            f"state_dict 不匹配：missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected


# ---------------------------------------------------------------------------
# 逐层 offload（超大组件内存常驻 + forward 时逐层上卡）
# ---------------------------------------------------------------------------

def _sequential_offload(module, device, name):
    """逐层 CPU offload（自写 forward hook）：forward 时逐层上卡、用完即卸。

    为什么不用 accelerate 的 cpu_offload_with_hook：它是**整模型** offload——
    pre_forward 把 26GB TE 整体搬上卡（+VAE 常驻 > 24GB 直接 OOM，真机实测），
    且官方 encoders.py 手动 hook.pre_forward 后调 text_encoder.model 子模块，
    绕过顶层 forward 的 post_forward，永远卸不回。
    自写逐层 hook：给每个带直接参数/buffer 的模块挂 pre（上卡）/post（指回
    原 CPU tensor），嵌套调用天然有序，峰值仅单层+激活。
    关键实现（真机 OOM 踩坑）：post 绝不能 `.to("cpu")` 新分配——推理权重
    只读，保存 pre 时的原 CPU tensor 引用，post 直接指回，CPU 侧零分配零
    拷贝；否则 4 步采样 = 256GB 分配循环，allocator 缓存只涨不还 OS，
    RSS 涨到 61GB 被 OOM killer 收尸。hook 挂在层上而非顶层——
    encoders.py 直调 text_encoder.model 也照走。
    """
    import torch as _t
    n = 0
    for sub in module.modules():
        if not any(p is not None for p in sub._parameters.values()) and \
           not any(b is not None for b in sub._buffers.values()):
            continue
        def _pre(mod, *_a, **_k):
            saved_p, saved_b = {}, {}
            for k, p in mod._parameters.items():
                if p is not None and p.device.type != "cuda":
                    saved_p[k] = p.data
                    p.data = p.data.to(device, non_blocking=True)
            for k, b in list(mod._buffers.items()):
                if b is not None and b.device.type != "cuda":
                    saved_b[k] = b
                    mod._buffers[k] = b.to(device, non_blocking=True)
            mod._offload_saved = (saved_p, saved_b)
        def _post(mod, *_a, **_k):
            saved_p, saved_b = getattr(mod, "_offload_saved", ({}, {}))
            for k, p in mod._parameters.items():
                if k in saved_p:
                    p.data = saved_p[k]
            for k, b in list(mod._buffers.items()):
                if k in saved_b:
                    mod._buffers[k] = saved_b[k]
            mod._offload_saved = ({}, {})
        sub.register_forward_pre_hook(_pre)
        sub.register_forward_hook(_post)
        n += 1
    _t.cuda.empty_cache() if device.type == "cuda" else None
    print(f"[quant-offload] {name}: 逐层 offload armed（{n} 个模块 hook）",
          flush=True)
    return module

# --- 内联结束 ---


OP_VERSION = "770c33e13741"
OP_FAMILY = "qwen_image"  # 必须等于本文件 stem（gate J 校验）

QWEN_DIR = os.environ.get("QWEN_IMAGE_DIR", "/models/qwen-image")


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)


# ---------------------------------------------------------------------------
# 公共 helper（与 wan.py 同款语义，保持各族一致）
# ---------------------------------------------------------------------------

def _models():
    from app.main import models
    return models


def _register(key, obj, est_bytes, arch, offload_mode="none",
              vram_bytes=None):
    """注册常驻对象（走 model_manager.register_external 统一口径）。

    est_bytes 是**权重体积**，不是显存需求：sequential offload 时权重在
    CPU、显存峰值只是「最大单层 + 激活」（20B 不能按 39GB 显存估算）。
    """
    models = _models()
    reg = getattr(models, "register_external", None)
    if reg is not None:
        return reg(key, obj, weights_bytes=est_bytes, arch=arch,
                   offload_mode=offload_mode, vram_bytes=vram_bytes)
    with models._lock:  # 旧服务端兜底
        models._evict_if_needed(est_bytes)
        models._pipes[key] = obj
        models._archs[key] = arch
        models._sizes[key] = est_bytes
        models._last_used[key] = time.time()
    return obj


def _require_file(path, what):
    """校验路径存在（文件或 Diffusers 目录）。"""
    if not os.path.exists(path):
        raise ValueError(f"qwen_image: 缺 {what}（{path}）")
    return path


def _size(p):
    """文件/目录体积。"""
    if not p:
        return 0
    if os.path.isdir(p):
        return sum(os.path.getsize(os.path.join(dp, f))
                   for dp, _, fs in os.walk(p) for f in fs)
    return os.path.getsize(p) if os.path.isfile(p) else 0


def _resolve(subdir, name):
    """QWEN_DIR 下解析权重路径：文件与目录皆可、支持绝对路径、裸名。"""
    if not name:
        return ""
    if os.path.isabs(name):
        return name
    cand = os.path.join(QWEN_DIR, subdir, name)
    if os.path.exists(cand):
        return cand
    cand = os.path.join(QWEN_DIR, name)
    if os.path.exists(cand):
        return cand
    return os.path.join(QWEN_DIR, subdir, name)  # 让 _require_file 报错


def _model_root(model):
    """官方 Diffusers 仓库布局根目录（<QWEN_DIR>/<model>/）。"""
    return os.path.join(QWEN_DIR, model or "Qwen-Image-2512")


def _dtype(dtype, default="bf16"):
    import torch
    d = (dtype or default).lower()
    if d in ("bf16", "bfloat16", "auto"):
        return torch.bfloat16
    if d in ("fp16", "float16", "half"):
        return torch.float16
    if d in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"qwen_image: dtype 须为 bf16/fp16/fp32，得 {dtype}")


def _offload(module, device, offload, name):
    """逐层 offload + _execution_device sentinel（与 H3/Wan 同款）。

    sentinel 必要性：diffusers 靠组件 _hf_hook.execution_device 推断执行
    设备；没有它会回落 cpu → 输入留 cpu 撞已上卡权重。
    """
    if device.type != "cuda":
        return module
    mode = str(offload or "sequential").lower()
    if mode in ("cuda", "full"):
        try:
            module.to(device)
            print(f"[qwen-image] {name} 全量上卡（offload=cuda）", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[qwen-image] {name} 全量上卡失败({type(e).__name__}) → "
                  f"退回 sequential", flush=True)
            module = _sequential_offload(module, device, name)
        return module
    if mode != "sequential":
        return module  # none：不挂假 sentinel（Wan animate2 教训）
    module = _sequential_offload(module, device, name)

    class _ExecDeviceSentinel:
        execution_device = "cuda"
    module._hf_hook = _ExecDeviceSentinel()
    return module


# ---------------------------------------------------------------------------
# 组件构建
# ---------------------------------------------------------------------------

def _build_transformer(model, transformer, dtype):
    """transformer 加载：官方 Diffusers 目录优先；Comfy 单文件尽力转换。

    - 目录（<QWEN_DIR>/<model>/transformer）：from_pretrained（bf16 官方
      9 分片，39GB）——**主路线，已验证布局**。
    - 单文件（diffusion_models/*.safetensors，Comfy-Org bf16/fp8_e4m3fn）：
      from_single_file；fp8 带 scale_weight 的 Comfy 布局若 metadata 不足
      转换失败，明确报错建议目录路线（不做静默错加载）。
    """
    from diffusers import QwenImageTransformer2DModel

    dt = _dtype(dtype)
    if transformer:
        path = _resolve("diffusion_models", transformer)
        _require_file(path, "transformer")
        if os.path.isdir(path):
            tr = QwenImageTransformer2DModel.from_pretrained(
                path, torch_dtype=dt)
        else:
            cfg_dir = os.path.join(_model_root(model), "transformer")
            cfg = cfg_dir if os.path.isdir(cfg_dir) else None
            try:
                tr = QwenImageTransformer2DModel.from_single_file(
                    path, config=cfg, torch_dtype=dt)
            except Exception as e:  # noqa: BLE001
                raise ValueError(
                    f"qwen_image: 单文件 {transformer} 转换失败"
                    f"（{type(e).__name__}: {e}）。fp8/Comfy 布局请改用官方"
                    f" Diffusers 目录（model 参数，如 Qwen-Image-2512）") from e
        src = path
    else:
        tr_dir = os.path.join(_model_root(model), "transformer")
        _require_file(tr_dir, f"transformer 目录（model={model}）")
        tr = QwenImageTransformer2DModel.from_pretrained(tr_dir,
                                                         torch_dtype=dt)
        src = tr_dir
    n_meta = sum(1 for p in tr.parameters() if p.device.type == "meta")
    if n_meta:
        raise ValueError(
            f"qwen_image: transformer 加载后仍有 {n_meta} 个 meta 参数——"
            f"权重未真正灌入（unexpected/missing keys 必查，FLUX 教训）")
    return tr, src


def _build_text_encoder(model, dtype):
    """Qwen2.5-VL-7B 文本编码器（**bf16**，fp16 会溢出）+ tokenizer。"""
    from transformers import (AutoTokenizer,
                              Qwen2_5_VLForConditionalGeneration)

    dt = _dtype(dtype, default="bf16")
    if dt.__str__() == "torch.float16":
        print("[qwen-image] ⚠️ TE 用 fp16 有溢出风险（T5 教训），建议 bf16",
              flush=True)
    root = _model_root(model)
    te_dir = os.path.join(root, "text_encoder")
    tok_dir = os.path.join(root, "tokenizer")
    _require_file(te_dir, "text_encoder 目录")
    _require_file(tok_dir, "tokenizer 目录")
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        te_dir, torch_dtype=dt)
    n_meta = sum(1 for p in te.parameters() if p.device.type == "meta")
    if n_meta:
        raise ValueError(
            f"qwen_image: text_encoder 加载后仍有 {n_meta} 个 meta 参数")
    tokenizer = AutoTokenizer.from_pretrained(tok_dir)
    return te, tokenizer, te_dir


def _build_vae(model, vae, device):
    """AutoencoderKLQwenImage（**fp32**，官方示例口径）。"""
    import torch
    from diffusers import AutoencoderKLQwenImage

    if vae:
        path = _resolve("vae", vae)
        _require_file(path, "vae")
        if os.path.isdir(path):
            m = AutoencoderKLQwenImage.from_pretrained(
                path, torch_dtype=torch.float32)
        else:
            cfg_dir = os.path.join(_model_root(model), "vae")
            m = AutoencoderKLQwenImage.from_single_file(
                path, config=cfg_dir if os.path.isdir(cfg_dir) else None)
        src = path
    else:
        vae_dir = os.path.join(_model_root(model), "vae")
        _require_file(vae_dir, "vae 目录")
        m = AutoencoderKLQwenImage.from_pretrained(
            vae_dir, torch_dtype=torch.float32)
        src = vae_dir
    if device != "cpu" and torch.cuda.is_available():
        m = m.to("cuda")
    return m, src


def _build_scheduler(model):
    """FlowMatchEuler：优先官方 scheduler 目录；缺失按 2512 config 构造
    （shift_terminal=0.02 / use_dynamic_shifting 等关键字段一个不能丢——
    FLUX scheduler 静默丢配置出黄色噪声的教训）。"""
    from diffusers import FlowMatchEulerDiscreteScheduler

    sch_dir = os.path.join(_model_root(model), "scheduler")
    if os.path.isdir(sch_dir) and os.path.isfile(
            os.path.join(sch_dir, "scheduler_config.json")):
        return FlowMatchEulerDiscreteScheduler.from_pretrained(sch_dir)
    print("[qwen-image] scheduler 目录缺失，按官方 2512 config 构造",
          flush=True)
    return FlowMatchEulerDiscreteScheduler(
        base_image_seq_len=256, base_shift=0.5, max_image_seq_len=8192,
        max_shift=0.9, num_train_timesteps=1000, shift=1.0,
        shift_terminal=0.02, time_shift_type="exponential",
        use_dynamic_shifting=True)


def _load_loras(pipe, spec, default_scale):
    """挂 LoRA（逗号分隔多挂，name@strength 语法；diffusers 格式）。

    Lightning LoRA（lightx2v）是标准 diffusers LoRA，直接
    load_lora_weights；多挂用 adapter_name + set_adapters。
    """
    items = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "@" in part:
            name, s = part.rsplit("@", 1)
            try:
                strength = float(s)
            except ValueError:
                name, strength = part, float(default_scale)
        else:
            name, strength = part, float(default_scale)
        items.append((name, strength))
    if not items:
        return pipe
    names, scales = [], []
    for i, (name, strength) in enumerate(items):
        path = _resolve("loras", name)
        _require_file(path, f"LoRA（{name}）")
        ad = f"lora{i}"
        pipe.load_lora_weights(os.path.dirname(path) or ".",
                               weight_name=os.path.basename(path),
                               adapter_name=ad)
        names.append(ad)
        scales.append(strength)
        print(f"[qwen-image] LoRA 挂载 {name}@{strength}", flush=True)
    if len(names) > 1:
        pipe.set_adapters(names, weights=scales)
    elif scales[0] != 1.0:
        pipe.set_adapters(names, weights=scales)
    return pipe


# ---------------------------------------------------------------------------
# 算子注册
# ---------------------------------------------------------------------------

def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "qwen_image.transformer_load",
        inputs={"model": "STRING", "transformer": "STRING",
                "dtype": "STRING", "offload": "STRING"},
        outputs={"transformer": "TRANSFORMER"},
        description="Qwen-Image transformer 组件加载（20B MMDiT）。官方 "
                    "Diffusers 目录（model 参数）或 Comfy 单文件（transformer "
                    "参数）。bf16 ≈39GB 走逐层 offload。")
    def _op_transformer_load(model="Qwen-Image-2512", transformer="",
                             dtype="bf16", offload="sequential"):
        import torch

        key = f"qwen-image:tr:{model}|{transformer}|{dtype}|{offload}"
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"transformer": models._pipes[key]}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr, src = _build_transformer(model, transformer, dtype)
        tr = _offload(tr, device, offload, "transformer")
        _budget_mode = "sequential" if str(offload).lower() in ("none", "") \
            else offload
        return {"transformer": _register(
            key, {"model": tr}, _size(src), "QwenImageTransformer2DModel",
            offload_mode=_budget_mode)}

    @registry.register(
        "qwen_image.te_load",
        inputs={"model": "STRING", "dtype": "STRING", "offload": "STRING"},
        outputs={"text_encoder": "TEXT_ENCODER"},
        description="Qwen-Image 文本编码器（Qwen2.5-VL-7B，bf16 ≈16.5GB）"
                    "组件加载 + Qwen2Tokenizer。")
    def _op_te_load(model="Qwen-Image-2512", dtype="bf16",
                    offload="sequential"):
        import torch

        key = f"qwen-image:te:{model}|{dtype}|{offload}"
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"text_encoder": models._pipes[key]}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        te, tokenizer, src = _build_text_encoder(model, dtype)
        te = _offload(te, device, offload, "text_encoder")
        _budget_mode = "sequential" if str(offload).lower() in ("none", "") \
            else offload
        return {"text_encoder": _register(
            key, {"model": te, "tokenizer": tokenizer}, _size(src),
            "Qwen2_5_VLForConditionalGeneration", offload_mode=_budget_mode)}

    @registry.register(
        "qwen_image.vae_load",
        inputs={"model": "STRING", "vae": "STRING", "device": "STRING"},
        outputs={"vae": "VAE"},
        description="Qwen-Image VAE 加载（AutoencoderKLQwenImage，**fp32**"
                    " ~243MB，常驻上卡）。")
    def _op_vae_load(model="Qwen-Image-2512", vae="", device="cuda"):
        key = f"qwen-image:vae:{model}|{vae}"
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"vae": models._pipes[key]}

        m, src = _build_vae(model, vae, device)
        return {"vae": _register(key, {"vae": m}, _size(src),
                                 "AutoencoderKLQwenImage")}

    @registry.register(
        "qwen_image.pipe",
        inputs={"transformer": "TRANSFORMER", "text_encoder": "TEXT_ENCODER",
                "vae": "VAE", "model": "STRING", "lora": "STRING",
                "lora_scale": "FLOAT", "max_sequence_length": "INT"},
        outputs={"model": "MODEL"},
        description="Qwen-Image 管道组装：组件 → QwenImagePipeline（+ 可选 "
                    "Lightning 等 LoRA，name@strength 逗号多挂）。组件常驻，"
                    "换 LoRA/参数不必重装 39GB transformer。")
    def _op_pipe(transformer=None, text_encoder=None, vae=None,
                 model="Qwen-Image-2512", lora="", lora_scale=1.0,
                 max_sequence_length=512):
        from diffusers import QwenImagePipeline

        if transformer is None or text_encoder is None or vae is None:
            raise ValueError("qwen_image.pipe: transformer / text_encoder / "
                             "vae 三个组件必填")
        tr_info, te_info, vae_info = transformer, text_encoder, vae
        key = (f"qwen-image:pipe:{id(tr_info.get('model')):x}|"
               f"{id(te_info.get('model')):x}|{id(vae_info.get('vae')):x}|"
               f"{model}|{lora}|{lora_scale}|{max_sequence_length}")
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"model": models._pipes[key]}

        pipe = QwenImagePipeline(
            scheduler=_build_scheduler(model),
            vae=vae_info["vae"],
            text_encoder=te_info["model"],
            tokenizer=te_info["tokenizer"],
            transformer=tr_info["model"])
        if lora:
            pipe = _load_loras(pipe, lora, lora_scale)
        pipe._qwen_max_seq = int(max_sequence_length)
        print(f"[qwen-image] pipe assembled（model={model}"
              f"{', lora=' + lora if lora else ''}）", flush=True)
        return {"model": _register(key, pipe, 0, "QwenImagePipeline")}

    @registry.register(
        "qwen_image.gen",
        inputs={"model": "MODEL", "prompt": "STRING",
                "negative_prompt": "STRING", "steps": "INT",
                "true_cfg_scale": "FLOAT", "width": "INT", "height": "INT",
                "seed": "INT", "max_sequence_length": "INT",
                "num_images": "INT"},
        outputs={"image": "IMAGE", "seed": "INT"},
        description="Qwen-Image T2I 生成。true_cfg_scale>1 且给了负提示才走 "
                    "CFG（官方默认 4.0）；挂 Lightning LoRA 时用 steps=4/8 + "
                    "true_cfg_scale=1.0。宽高须 16 的倍数（自动对齐）。")
    def _op_gen(model, prompt="", negative_prompt="", steps=50,
                true_cfg_scale=4.0, width=1328, height=1328, seed=-1,
                max_sequence_length=0, num_images=1):
        import random

        import numpy as np
        import torch
        from PIL import Image

        pipe = model
        if not hasattr(pipe, "transformer"):
            raise ValueError("qwen_image.gen: model 不是 qwen_image.pipe 的"
                             "输出")
        if not prompt:
            raise ValueError("qwen_image.gen: prompt 必填")

        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        # VAE 8 × patch 2 → 16 对齐（官方 aspect_ratio_resize 同款）
        mod = 16
        h = max(mod, round(int(height) / mod) * mod)
        w = max(mod, round(int(width) / mod) * mod)
        if (h, w) != (int(height), int(width)):
            print(f"[qwen-image] {width}x{height} -> {w}x{h}（16 对齐）",
                  flush=True)

        msl = int(max_sequence_length) or int(
            getattr(pipe, "_qwen_max_seq", 512))
        kwargs = dict(prompt=prompt,
                      negative_prompt=negative_prompt or None,
                      height=h, width=w,
                      num_inference_steps=int(steps),
                      true_cfg_scale=float(true_cfg_scale),
                      num_images_per_prompt=max(1, int(num_images)),
                      max_sequence_length=msl,
                      generator=gen, output_type="np")
        t0 = time.time()
        out = pipe(**kwargs)
        arr = np.asarray(out.images[0])  # (N,H,W,3) [0,1] → 取首张前判型
        if arr.ndim == 3:
            arr = arr[None]
        if arr.shape[-1] not in (1, 3) and arr.shape[1] in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.min() < -0.05:
            arr = (arr + 1.0) / 2.0
        images = [Image.fromarray(
            (np.clip(f, 0, 1) * 255).round().astype("uint8")) for f in arr]
        print(f"[qwen-image] gen done {len(images)}张 {w}x{h} "
              f"steps={int(steps)} cfg={float(true_cfg_scale)} "
              f"{time.time()-t0:.1f}s", flush=True)
        return {"image": images[0] if len(images) == 1 else images,
                "seed": seed}
