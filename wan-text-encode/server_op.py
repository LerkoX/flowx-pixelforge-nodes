"""Wan 算子族：通义万相视频生成（Wan2.1 14B / Wan2.2 MoE）+ Wan-Animate 角色替换。

随节点包自注册（与 minimax_h3 同款模式）。

模型事实（对拍 diffusers 0.42.1 内置实现）：
- 管道三分：
  * WanPipeline               T2V
  * WanImageToVideoPipeline   I2V（给 last_image 即首尾帧 FLF2V）
  * WanAnimatePipeline        角色动画 / **角色替换**（image + pose/face 视频
                              [+ background_video + mask_video]，mode=animate|replace）
- **与 MiniMax-H3 的关键差异（移植代码时必须逐条对齐）**：
  1. **Wan 要 CFG**：guidance_scale>1 时每步两次前向（cond+uncond），
     negative_prompt 生效；H3 是 guidance-distilled，无 CFG 无负提示。
  2. **帧数契约 4k+1**（81/121/161…），H3 是 17n+5。
  3. **fps**：Wan2.1 全系 16fps；Wan2.2-TI2V-5B 是 24fps。
  4. **VAE 必须 fp32**（官方明确：fp32 解码质量更好），不能像 H3 那样
     bf16 直跑；Wan VAE 体积小（~500MB fp32）不构成显存压力。
  5. **Wan2.2 是 MoE 双去噪器**：transformer（高噪）+ transformer_2（低噪），
     boundary_ratio≈0.875 处切换；LoRA 默认只挂第一个，需
     load_into_transformer_2=True 才同时挂。
  6. Wan **不生成音频**（H3 是联合音视频），音轨由节点侧 save-video 决定。

权重布局（WAN_DIR，默认 /models/wan；权重优先共享盘软链）：
- diffusion_models/<transformer>.safetensors   WanTransformer3DModel 单文件
  （Comfy-Org 重打包 bf16/fp8；diffusers from_single_file 直读，免 ComfyUI 依赖）
- diffusion_models/<transformer2>.safetensors  Wan2.2 低噪专家（可选）
- text_encoders/<umt5>.safetensors             UMT5-XXL（Comfy-Org
                                               umt5_xxl_*.safetensors 单文件）
- vae/<vae>.safetensors                        AutoencoderKLWan（fp32）
- image_encoders/<clip_vision>.safetensors     CLIP-ViT-H（I2V/Animate 需要）
- loras/<lora>.safetensors                     LightX2V 4步加速等
- config/                                      官方 diffusers config 目录
  （transformer/ transformer_2/ vae/ text_encoder/ tokenizer/
   scheduler/ image_encoder/，KB 级；from_single_file 缺 metadata 时用）

显存/内存相位（24GB 卡跑 14B——与 H3 同一套解法但量级小得多）：
- transformer 14B bf16 ≈ 28GB > 24GB ⇒ 逐层 offload（_sequential_offload，
  CPU 常驻，forward 时逐层上卡，峰值仅单层+激活）
- text_encoder UMT5-XXL bf16 ≈ 11GB（fp8 scaled ≈ 5.6GB）⇒ 同样逐层 offload
- vae fp32 ≈ 0.5GB ⇒ 常驻上卡
- 合计 CPU 常驻 ≈ 39GB << 64GB 内存（H3 是 58GB，Wan 宽裕得多）

算子（拆分式，对齐 SD 的 checkpoint-loader → clip-text-encode → ksampler）：
- wan.transformer_load / te_load / vae_load / image_encoder_load → 组件对象
- wan.pipe(task, transformer, transformer2?, text_encoder, vae, image_encoder?,
  boundary_ratio, flow_shift, ...) → MODEL
- wan.encode(model, prompt, negative_prompt, max_sequence_length) → COND
  （prompt_embeds + negative_prompt_embeds；Wan 有 CFG，正负两路）
- wan.gen(model, cond?, prompt?, negative_prompt?, image?, last_image?,
  num_frames, height, width, steps, guidance_scale, seed, flow_shift)
  → VIDEO（frames+fps，无音轨）+ seed；分钟级任务走 POST /jobs
- wan.animate(model, image, pose_video, face_video, background_video?,
  mask_video?, prompt, negative_prompt, mode, segment_frame_length,
  guidance_scale, seed) → VIDEO
"""

import math
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


OP_VERSION = "83efe3d69954"
OP_FAMILY = "wan"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py

class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)


WAN_DIR = os.environ.get("WAN_DIR", "/models/wan")
WAN_CONFIG_DIR = os.environ.get("WAN_CONFIG_DIR", os.path.join(WAN_DIR, "config"))

# fps：Wan2.1 全系 16；Wan2.2-TI2V-5B 是 24（按 transformer 文件名自动判定，
# 可用 fps 参数覆盖）
WAN_FPS_DEFAULT = 16
WAN_FPS_TI2V5B = 24

# 各模型族的 transformer 结构参数（from_single_file 缺 metadata 时的兜底
# config；取自 Wan-AI 官方 Diffusers 仓库 config.json）
_TRANSFORMER_CFG = {    # Wan2.1 T2V/I2V/FLF2V 14B
    "14b": {
        "_class_name": "WanTransformer3DModel",
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "dim": 5120,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_channels": 16,
        "num_attention_heads": 40,
        "num_layers": 40,
        "out_channels": 16,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
    },
    # Wan2.1 T2V 1.3B
    "1.3b": {
        "_class_name": "WanTransformer3DModel",
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "dim": 1536,
        "eps": 1e-06,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "in_channels": 16,
        "num_attention_heads": 12,
        "num_layers": 30,
        "out_channels": 16,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
    },
    # Wan2.2 T2V/I2V A14B（MoE 单专家形状与 14B 同构）
    "2.2-14b": {
        "_class_name": "WanTransformer3DModel",
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "dim": 5120,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_channels": 16,
        "num_attention_heads": 40,
        "num_layers": 40,
        "out_channels": 16,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
    },
    # Wan2.2 TI2V 5B
    "5b": {
        "_class_name": "WanTransformer3DModel",
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "dim": 3072,
        "eps": 1e-06,
        "ffn_dim": 14336,
        "freq_dim": 256,
        "in_channels": 48,
        "num_attention_heads": 24,
        "num_layers": 30,
        "out_channels": 48,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
    },
}

# Wan-VACE 专用结构（与普通 Wan transformer 的差异）：
# - in_channels=96：VACE 把「原视频 + 遮罩 + 参考」编成条件通道拼进来
# - added_kv_proj_dim 非空：控制信号走额外 KV 投影
# 数值取自 Wan2.1-VACE-14B-Diffusers；1.3B 版按 dim/heads/layers 缩放。
_VACE_CFG = {"14b": {
        "_class_name": "WanVACETransformer3DModel",
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "dim": 5120,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_channels": 16,
        "num_attention_heads": 40,
        "num_layers": 40,
        "out_channels": 16,
        "vace_in_channels": 96,
        "vace_layers": [0, 5, 10, 15, 20, 25, 30, 35],
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
    },
    "1.3b": {
        "_class_name": "WanVACETransformer3DModel",
        "added_kv_proj_dim": None,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "dim": 1536,
        "eps": 1e-06,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "in_channels": 16,
        "num_attention_heads": 12,
        "num_layers": 30,
        "out_channels": 16,
        "vace_in_channels": 96,
        "vace_layers": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28],
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
    },
}


_VAE_CFG = {
    "_class_name": "AutoencoderKLWan",
    "base_dim": 96,
    "decoder_base_dim": 256,
    "dim_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attn_scales": [],
    "temperal_downsample": [False, True, True],
    "dropout": 0.0,
    "z_dim": 16,
    "scale_factor_temporal": 4,
    "scale_factor_spatial": 8,
    "latents_mean": [
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ],
    "latents_std": [
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ],
}


# ---------------------------------------------------------------------------
# 组件装配基础
# ---------------------------------------------------------------------------

def _models():
    from app.main import models
    return models


def _register(key, obj, est_bytes, arch, offload_mode="none",
              vram_bytes=None):
    """注册常驻对象（走 model_manager.register_external 统一口径）。

    est_bytes 是**权重体积**（文件/state_dict），不是显存需求：
    offload=sequential 时权重在 CPU、显存峰值只是「最大单层 + 激活」，
    服务端按 vram_budget_of 折算（否则 14B 会被当成 28GB 显存需求而误拒）。
    """
    models = _models()
    reg = getattr(models, "register_external", None)
    if reg is not None:
        return reg(key, obj, weights_bytes=est_bytes, arch=arch,
                   offload_mode=offload_mode, vram_bytes=vram_bytes)
    # 旧服务端兜底（无 register_external）：维持原语义
    with models._lock:
        models._evict_if_needed(est_bytes)
        models._pipes[key] = obj
        models._archs[key] = arch
        models._sizes[key] = est_bytes
        models._last_used[key] = time.time()
    return obj
def _require_file(path, what):
    """校验路径存在（**文件或 Diffusers 目录**）。

    目录 = from_pretrained 的仓库/组件目录（官方 *-Diffusers 发布格式）；
    文件 = Comfy-Org 单文件 safetensors。两者都是合法权重来源。
    """
    if not os.path.exists(path):
        raise ValueError(f"wan: 缺 {what}（{path}）")
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
    """WAN_DIR 下解析权重路径：支持**文件与目录**、绝对路径、裸名。

    目录 = Diffusers 格式（config.json + 分片权重），走 from_pretrained；
    文件 = 单文件 safetensors，走 from_single_file。VACE 官方发布的就是
    目录格式（ModelScope/HF 的 *-Diffusers 仓库），单文件只覆盖 Comfy 生态。
    """
    if not name:
        return ""
    # 本项目的目录命名（Comfy 风格）与 Diffusers 仓库的子目录名不同，做别名映射：
    #   我们: diffusion_models / text_encoders / vae / image_encoders
    #   Diffusers: transformer / text_encoder / vae / image_encoder
    _ALIAS = {"diffusion_models": ("diffusion_models", "transformer",
                                   "transformer_2"),
              "text_encoders": ("text_encoders", "text_encoder"),
              "vae": ("vae",),
              "image_encoders": ("image_encoders", "image_encoder")}
    subs = _ALIAS.get(subdir, (subdir,))

    def _pick(base):
        """base 下按别名顺序找组件子目录（必须含 config.json 或分片权重）。"""
        for sname in subs:
            cand = os.path.join(base, sname)
            if os.path.isdir(cand):
                return cand
        return None

    if os.path.isabs(name):
        if os.path.exists(name):
            # 绝对路径可能是组件目录本身，也可能是仓库根 → 有 config.json 就用它
            if os.path.isdir(name) and not os.path.isfile(
                    os.path.join(name, "config.json")):
                sub = _pick(name)
                if sub:
                    return sub
            return name
        return name

    # ① 本项目布局：WAN_DIR/<Comfy 子目录>/<name>
    for sname in subs:
        cand = os.path.join(WAN_DIR, sname, name)
        if os.path.exists(cand):
            return cand
    # ② WAN_DIR/<name>（组件直接放根）
    cand = os.path.join(WAN_DIR, name)
    if os.path.exists(cand):
        if os.path.isdir(cand) and not os.path.isfile(
                os.path.join(cand, "config.json")):
            sub = _pick(cand)
            if sub:
                return sub
        return cand
    # ③ 仓库布局：MODELS_DIR/<name>/<Diffusers 子目录>（如 wan-vace-1.3b/transformer）
    root = os.path.dirname(WAN_DIR)
    repo = os.path.join(root, name)
    if os.path.isdir(repo):
        sub = _pick(repo)
        return sub or repo
    return os.path.join(WAN_DIR, subdir, name)


def _guess_arch(name):
    """从 transformer 文件名猜结构档位（14b / 1.3b / 2.2-14b / 5b）。"""
    low = name.lower()
    if "ti2v" in low and "5b" in low:
        return "5b"
    if "2.2" in low or "2_2" in low:
        return "2.2-14b"
    if "1.3b" in low or "1_3b" in low:
        return "1.3b"
    return "14b"


def _guess_fps(transformer_name):
    low = (transformer_name or "").lower()
    return WAN_FPS_TI2V5B if ("ti2v" in low and "5b" in low) else WAN_FPS_DEFAULT


def _config_json(name):
    """读 WAN_CONFIG_DIR/<name>/ 下的 config；不存在返回 None。

    ⚠ 文件名兼容（2026-11-14 真机大坑）：diffusers 落盘的 scheduler 配置
    叫 **scheduler_config.json** 而不是 config.json——只找 config.json 会
    静默拿到 None → 调用方 `or {}` 用全默认参数建 UniPC
    （prediction_type=epsilon），flow-matching 模型直接出黄色噪声，
    且日志只打 flow_shift 毫无异常。此类"静默错配置"必须防住。
    """
    import json
    d = os.path.join(WAN_CONFIG_DIR, name)
    for fn in ("config.json", f"{name}_config.json"):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


def _offload(module, device, offload, name, est_gb=None):
    """逐层 offload + _execution_device sentinel（与 H3 同款）。

    sentinel 必要性：diffusers/transformers 靠组件 _hf_hook.execution_device
    推断执行设备；没有它会回落 pipe.device=cpu → 输入留在 cpu 撞已上卡的
    权重（index_select 设备冲突）。无 pre_forward 属性，安全跳过。
    """
    if device.type != "cuda":
        return module
    # **offload="cuda" = 全量常驻 GPU（2026-11-16 真机）**：官方 VACE 示例
    # 就是 `pipe.to("cuda")`，1.3B(6.7GB) 在 24G 卡上完全放得下，且**只有
    # 这条路能保证数值正确**（挂 sequential hook 时 vace_blocks 的流式搬运
    # 会让输出退化成彩色噪声）。放不下时自动退回 sequential。
    if str(offload).lower() in ("cuda", "full"):
        try:
            module.to(device)
            print(f"[wan] {name} 全量上卡（offload=cuda）", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[wan] {name} 全量上卡失败({type(e).__name__}) → 退回"
                  f" sequential", flush=True)
            module = _sequential_offload(module, device, name)
        return module
    if offload != "sequential":
        # **offload=none 时不要挂假 `_hf_hook`**（2026-11-13 真机）：
        # 这个 sentinel 声称 execution_device=cuda，却没有任何搬运能力；
        # 后续 `apply_group_offloading`（在 wan.animate2 里挂 block 级 hook）
        # 会因此被误导 —— 组件被当作"已在 GPU"，等真正 onload 时显存已被占满
        # （实测 23.49GiB/23.54GiB，只剩 40MB → OOM）。
        # 交给 diffusers 自己的 hook 体系判断即可。
        return module
    module = _sequential_offload(module, device, name)

    class _ExecDeviceSentinel:
        execution_device = "cuda"
    module._hf_hook = _ExecDeviceSentinel()
    return module


# ---------------------------------------------------------------------------
# transformer
# ---------------------------------------------------------------------------

def _build_transformer(path, dtype, which="transformer"):
    """单文件 safetensors → WanTransformer3DModel（bf16/fp8 单文件直读）。

    from_single_file 的 config 来源优先级：
      ① WAN_CONFIG_DIR/<which>/config.json（官方 Diffusers 目录 copy，最准）
      ② safetensors 自带 metadata（Comfy-Org 重打包件通常带）
      ③ 结构兜底表（按文件名猜 14b/1.3b/5b）
    三者都拿不到才报错。
    """
    import torch
    from diffusers import WanTransformer3DModel

    t0 = time.time()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    # ⚠ diffusers 的 from_single_file(config=...) **只接受路径字符串**（repo id
    # 或本地 Diffusers 目录），传 dict 会报 "Invalid `config` argument"
    # （真机踩坑，和 FLUX VAE 同款）。故优先用 WAN_CONFIG_DIR/<which> 目录。
    # config 目录优先级：
    #   ① WAN_CONFIG_DIR/<which>（官方 Diffusers 仓库 copy，最准）
    #   ② 内置结构表写临时目录（Comfy-Org 单文件用 cross_attn.k/q/o 命名，**必须**
    #      走 from_single_file 的 key 转换表；而 config 参数只收路径 → 落临时盘）
    #   ③ 都没有：报错（不许静默建空模型，否则权重语义错乱 → 解码出抽象色块）
    cfg_dir = os.path.join(WAN_CONFIG_DIR, which)
    if not os.path.isfile(os.path.join(cfg_dir, "config.json")):
        import json as _json
        import tempfile
        arch = _guess_arch(os.path.basename(path))
        print(f"[wan] {which} 无 config 目录，用内置 {arch} 结构写临时 config",
              flush=True)
        cfg_dir = tempfile.mkdtemp(prefix=f"wan_cfg_{which}_")
        with open(os.path.join(cfg_dir, "config.json"), "w",
                  encoding="utf-8") as f:
            _json.dump(_TRANSFORMER_CFG[arch], f)
    print(f"[wan] {which} config 目录 {cfg_dir}", flush=True)
    model = WanTransformer3DModel.from_single_file(
        path, config=cfg_dir, torch_dtype=torch_dtype)
    n = sum(p.numel() for p in model.parameters())
    print(f"[wan] {which} ready {n/1e9:.2f}B params "
          f"{time.time()-t0:.1f}s", flush=True)
    return model


# --- Relighting LoRA（Wan2.2 官方 replace 专用光照融合）---------------------
# 官方 Wan-AI/Wan2.2-Animate-14B/relighting_lora（rank=alpha=128 → scale=1），
# 官方键名 base_model.model.blocks.N.{self_attn,cross_attn,ffn}.*.{lora_A,lora_B}
# 与 diffusers WanAnimateTransformer3DModel 命名的映射表（2026-11-15 调研）：
_RELIGHT_KEY_MAP = {
    ("self_attn", "q"): "attn1.to_q",
    ("self_attn", "k"): "attn1.to_k",
    ("self_attn", "v"): "attn1.to_v",
    ("self_attn", "o"): "attn1.to_out.0",
    ("cross_attn", "q"): "attn2.to_q",
    ("cross_attn", "k"): "attn2.to_k",
    ("cross_attn", "v"): "attn2.to_v",
    ("cross_attn", "o"): "attn2.to_out.0",
    ("cross_attn", "k_img"): "attn2.add_k_proj",
    ("cross_attn", "v_img"): "attn2.add_v_proj",
    ("ffn", "0"): "ffn.net.0.proj",
    ("ffn", "2"): "ffn.net.2",
}


def _merge_relight_lora(transformer, lora_path, scale=1.0):
    """把 Relighting LoRA 直接合并进 transformer 权重（幂等）。

    官方注释：relighting lora for character replacement——让替换人物的
    光照匹配背景环境（治"人物像另一个图层"）。merge 而非 peft 挂载：
    一次性、无运行时 hook、与 group_offloading 无交互（必须在挂 offload
    hook 之前、权重还在 CPU 内存态时做）。幂等标记 _relight_merged。
    """
    import re
    import torch
    if getattr(transformer, "_relight_merged", False):
        print("[wan] relight lora 已合并（组件复用，跳过）", flush=True)
        return
    if not os.path.isfile(lora_path):
        raise FileNotFoundError(f"relight lora 不存在: {lora_path}")
    from safetensors.torch import load_file
    t0 = time.time()
    sd = load_file(lora_path)  # CPU
    pat = re.compile(
        r"base_model\.model\.blocks\.(\d+)\.(self_attn|cross_attn|ffn)\."
        r"(q|k|v|o|k_img|v_img|0|2)\.lora_(A|B)\.weight")
    pairs = {}
    for k, v in sd.items():
        m = pat.match(k)
        if not m:
            continue
        blk, att, tgt, ab = m.group(1), m.group(2), m.group(3), m.group(4)
        diff_name = _RELIGHT_KEY_MAP.get((att, tgt))
        if diff_name is None:
            continue
        full = f"blocks.{blk}.{diff_name}"
        pairs.setdefault(full, {})[ab] = v
    n_merged, n_missing = 0, 0
    with torch.no_grad():
        for full, ab in pairs.items():
            if "A" not in ab or "B" not in ab:
                continue
            # 逐级 getattr 定位 Linear
            mod = transformer
            try:
                for part in full.split("."):
                    mod = mod[int(part)] if part.isdigit() \
                        else getattr(mod, part)
            except AttributeError:
                n_missing += 1
                continue
            delta = (ab["B"].float() @ ab["A"].float()) * float(scale)
            if delta.shape != mod.weight.shape:
                print(f"[wan][relight] 形状不符 {full}: "
                      f"{tuple(delta.shape)} vs {tuple(mod.weight.shape)}",
                      flush=True)
                n_missing += 1
                continue
            mod.weight.data += delta.to(mod.weight.dtype)
            n_merged += 1
    if n_merged == 0:
        raise RuntimeError(f"relight lora 合并 0 项（键名映射失效）: "
                           f"{lora_path}")
    transformer._relight_merged = True
    print(f"[wan] relight lora 合并完成 {n_merged} 项"
          f"{'（跳过 ' + str(n_missing) + '）' if n_missing else ''} "
          f"({time.time() - t0:.1f}s)", flush=True)


def _build_animate2_transformer(path, dtype, which="transformer"):
    """Wan-Animate transformer（**WanAnimatePipeline 路线**，2026-11-14 重构）。

    ⚠ 历史纠错：本函数原先加载 `WanAnimate2Transformer3DModel`（diffusers
    0.42 给下一代模型准备的新架构），用 Wan2.2-Animate-14B 权重时有
    **138 个 unexpected keys**（pose_patch_embedding/face_adapter/motion_encoder
    全部没有槽位）——模型类选错了，不是权重问题。

    正解（真机验证）：Wan2.2-Animate-14B 配套的是
    **`WanAnimateTransformer3DModel`（无 "2"）** + `WanAnimatePipeline`
    （普通 pipeline，diffusers 0.36+ 就有），全模块齐全。

    权重：官方 diffusers 格式 `Wan-AI/Wan2.2-Animate-14B-Diffusers`
    （ModelScope 下载 fp32 65GB → 自写流式转换器离线转 bf16 33GB，
    逐片校验），落盘在 ANIMATE_DIFFUSERS_DIR。加载 <2 分钟，0 missing /
    0 unexpected。
    """
    import time as _time

    import torch
    from diffusers import WanAnimateTransformer3DModel

    ANIMATE_DIFFUSERS_DIR = os.environ.get(
        "WAN_ANIMATE_DIR", "/models/wan-animate-diffusers/transformer")
    t0 = _time.time()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]

    if not os.path.isdir(ANIMATE_DIFFUSERS_DIR):
        raise ValueError(
            f"wan: animate 权重目录不存在：{ANIMATE_DIFFUSERS_DIR}"
            f"（需 Wan-AI/Wan2.2-Animate-14B-Diffusers 的 transformer，"
            f"ModelScope 下载 + conv_bf16.py 转 bf16；"
            f"原始 Wan2.2-Animate-14B 是 ComfyUI 键名，diffusers 无转换器）")

    # **参数名必须是 torch_dtype（2026-11-16 雪花事故根因之一）**：
    # 曾写成 `dtype=`，diffusers 不认 → 静默按默认精度加载，加载耗时
    # 异常短（3090 上 18s vs 4090 上 90~120s），权重实际没进内存 →
    # transformer 输出"尺度正常但方向随机"的噪声 → 成片人物区一片彩色
    # 雪花（背景是原帧混回，所以完好）。参数名 + 下方硬校验双保险。
    model = WanAnimateTransformer3DModel.from_pretrained(
        ANIMATE_DIFFUSERS_DIR, torch_dtype=torch_dtype,
        low_cpu_mem_usage=True)
    # **加载完整性硬校验**：任何参数停在 meta / 抽样权重全 0 都视为
    # 加载失败，直接抛错——绝不静默跑出雪花片。
    _ps = list(model.parameters())
    _meta = sum(1 for _prm in _ps if _prm.device.type == "meta")
    if _meta:
        raise RuntimeError(
            f"wan: animate transformer 加载失败——{_meta}/{len(_ps)} 个参数"
            f"停在 meta（权重未真正载入）。检查 from_pretrained 的 "
            f"torch_dtype / low_cpu_mem_usage 与磁盘权重完整性。")
    _probe = dict(model.named_parameters()).get("blocks.0.attn1.to_q.weight")
    _probe_s = "n/a"
    if _probe is not None:
        _pv = _probe.detach().float().abs()
        _probe_s = f"mean={float(_pv.mean()):.5f} max={float(_pv.max()):.5f}"
        if float(_pv.max()) == 0.0:
            raise RuntimeError(
                "wan: animate transformer 权重全 0（加载失败或被清零），"
                "拒绝生成——请检查权重文件与加载路径。")
    n = sum(prm.numel() for prm in _ps) / 1e9
    print(f"[wan] animate transformer ready {_time.time()-t0:.1f}s "
          f"({n:.2f}B 参数, dtype={torch_dtype}, meta={_meta}, "
          f"抽样 blocks.0.attn1.to_q {_probe_s})", flush=True)
    return model


def _build_vace_transformer(path, dtype, which="transformer"):
    """Wan-VACE transformer（WanVACETransformer3DModel，与普通 Wan 结构不同）。

    差异：in_channels=96（原视频+遮罩+参考拼成条件通道）、added_kv_proj_dim
    非空（控制信号走额外 KV）。**不能用 WanTransformer3DModel 加载**。
    config 优先级同 _build_transformer：WAN_CONFIG_DIR/<which> → 内置表写临时目录。
    """
    import json as _json
    import tempfile
    import time as _time

    import torch
    from diffusers import WanVACETransformer3DModel

    t0 = _time.time()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    if os.path.isdir(path):
        # Diffusers 目录：先校验 config 是不是真 VACE（2026-11-16 真机坑）
        # ModelScope 上 Wan2.1-VACE-1.3B-Diffusers 的 transformer/config.json
        # 被落成了普通 Wan 结构（in_channels=16 / added_kv_proj_dim=None），
        # 照它装配 → 条件通道全废 → 出雪花。这里比对 in_channels，
        # 不合格就用内置 _VACE_CFG 表（按 dim 判 1.3b/14b）。
        # **VACE config 修正（2026-11-16 真机根因）**：ModelScope 的
        # Wan2.1-VACE-*-Diffusers 的 transformer/config.json 里
        # `added_kv_proj_dim = None`，而 diffusers 的 WanVACETransformer3DModel
        # 只有在 added_kv_proj_dim 非空时才构造 `vace_patch_embedding`
        # （96 通道）与 `vace_blocks`。为 None ⇒ 这些模块根本不建 ⇒ 权重全是
        # unexpected keys ⇒ 模型退化成普通 T2V（丢掉全部控制条件）⇒ 输出雪花。
        # 注意：`in_channels` 应保持 16（视频通道），VACE 条件走独立分支，
        # 不要再改成 96（那是早期误判）。
        _cfg_p = os.path.join(path, "config.json")
        if os.path.isfile(_cfg_p):
            try:
                _c = _json.load(open(_cfg_p, encoding="utf-8"))
                # **VACE 结构的正确字段（2026-11-16 真机实证）**：
                #   added_kv_proj_dim 必须保持 **None**（它控制 attn2 内的
                #     add_k_proj/add_v_proj；该 checkpoint 没有这些权重，
                #     设成 1536 会造出 225 个随机初始化参数 → 雪花）
                #   vace_layers      = 挂 vace_blocks 的主层索引（权重实测
                #     vace_blocks 0..14 + 主 blocks 30 ⇒ 每 2 层一个）
                #   vace_in_channels = 96（VACE 条件通道）
                #   注意 in_channels 保持 16（视频通道，双 patch_embedding）
                _nl = int(_c.get("num_layers") or 30)
                _missing_cfg = (
                    _c.get("added_kv_proj_dim") is not None
                    or not _c.get("vace_layers")
                    or not _c.get("vace_in_channels"))
                if _missing_cfg:
                    _c["added_kv_proj_dim"] = None
                    _c["vace_in_channels"] = 96
                    _c["vace_layers"] = list(range(0, _nl, 2))
                    cfg_dir = tempfile.mkdtemp(prefix="wan_vace_fix_")
                    with open(os.path.join(cfg_dir, "config.json"), "w",
                              encoding="utf-8") as f:
                        _json.dump(_c, f)
                    # 让它能从 cfg_dir 读到 config、从原目录读到权重：
                    # 用 from_pretrained(config=cfg_dir) 等价物 —— 直接把
                    # 校正后的 config 写进模型构造，权重仍从 path 加载。
                    print(f"[wan] VACE config 修正：vace_layers="
                          f"{_c['vace_layers'][:4]}...({len(_c['vace_layers'])}层) "
                          f"vace_in=96 added_kv=None", flush=True)
                    import torch as _t2
                    from safetensors.torch import load_file as _lf2
                    _m = WanVACETransformer3DModel.from_config(
                        _c, torch_dtype=torch_dtype)
                    _sd = {}
                    _idx_p = os.path.join(
                        path, "diffusion_pytorch_model.safetensors.index.json")
                    if os.path.isfile(_idx_p):
                        _idx = _json.load(open(_idx_p, encoding="utf-8"))
                        for _sh in sorted(set(_idx["weight_map"].values())):
                            _sd.update(_lf2(os.path.join(path, _sh)))
                    else:
                        for _fn in os.listdir(path):
                            if _fn.endswith(".safetensors"):
                                _sd.update(_lf2(os.path.join(path, _fn)))
                    _miss, _unexp = _m.load_state_dict(
                        _sd, strict=False, assign=True)
                    print(f"[wan] VACE 内置 config 装配："
                          f"missing={len(_miss)} unexpected={len(_unexp)}",
                          flush=True)
                    if _miss:
                        print(f"[wan] ⚠ missing 样例 {_miss[:6]}", flush=True)
                    if _unexp:
                        print(f"[wan] ⚠ unexpected 样例 {_unexp[:6]}",
                              flush=True)
                    del _sd
                    # **精度统一（2026-11-16）**：该 checkpoint 90% 是 F32，
                    # assign=True 会保留原 dtype → 与 pipeline 的 bf16 输入
                    # 冲突（Input type (float) and bias type (c10::BFloat16)
                    # should be the same）。统一转到目标 dtype，顺带省一半内存。
                    _m = _m.to(torch_dtype)
                    _n = sum(pp.numel() for pp in _m.parameters()) / 1e9
                    print(f"[wan] VACE {which} ready {_n:.2f}B "
                          f"{_time.time()-t0:.1f}s", flush=True)
                    return _m
            except Exception as _ce:       # noqa: BLE001
                print(f"[wan] ⚠ VACE config 修正失败"
                      f"（{type(_ce).__name__}: {str(_ce)[:100]}），"
                      f"回退 from_pretrained", flush=True)
        model = WanVACETransformer3DModel.from_pretrained(
            path, torch_dtype=torch_dtype)
        n = sum(p.numel() for p in model.parameters())
        print(f"[wan] VACE {which} from_pretrained {n/1e9:.2f}B "
              f"{_time.time()-t0:.1f}s", flush=True)
        return model
    cfg_dir = os.path.join(WAN_CONFIG_DIR, which)
    if not os.path.isfile(os.path.join(cfg_dir, "config.json")):
        arch = "1.3b" if "1.3b" in os.path.basename(path).lower() else "14b"
        print(f"[wan] VACE {which} 用内置 {arch} 结构写临时 config", flush=True)
        cfg_dir = tempfile.mkdtemp(prefix=f"wan_vace_{which}_")
        with open(os.path.join(cfg_dir, "config.json"), "w",
                  encoding="utf-8") as f:
            _json.dump(_VACE_CFG[arch], f)
    model = WanVACETransformer3DModel.from_single_file(
        path, config=cfg_dir, torch_dtype=torch_dtype)
    # **必须把 keep-in-fp32 模块转回 fp32（2026-11-16 真机定位）**：
    # from_single_file 走的是 `model.to(dtype)` 式的整体 cast，diffusers 会
    # 警告 "Casting directly with to() can lead to inconsistent results"。
    # rope / time_embedder / scale_shift_table / norm1..3 在 fp16-bf16 下会
    # 溢出或精度失真（time_embedder 的 sinusoidal 频率、norm 的统计量），
    # 后果就是**整幅画面变成彩色噪声**（看起来像雪花，其实数值全错）。
    # from_pretrained 路径由 _keep_in_fp32_modules 自动处理，单文件路径没有。
    _KF = ("rope", "time_embedder", "scale_shift_table",
           "norm1", "norm2", "norm3")
    _n32 = 0
    for _name, _mod in model.named_modules():
        if _name.split(".")[-1] in _KF:
            _mod.to(torch.float32)
            _n32 += 1
    for _pn, _pp in model.named_parameters():
        if _pn.split(".")[-2:-1] and _pn.split(".")[-2] in _KF:
            _pp.data = _pp.data.float()
    print(f"[wan] VACE {which} keep-fp32 模块已恢复 {_n32} 个",
          flush=True)
    n = sum(p.numel() for p in model.parameters())
    print(f"[wan] VACE {which} ready {n/1e9:.2f}B params "
          f"{_time.time()-t0:.1f}s", flush=True)
    return model


def _lora_unmatched(pipe, path):
    """检查 LoRA 文件的层名能否映射到目标 transformer，返回未匹配名列表。

    diffusers 的 load_lora_weights 对无法映射的 key **静默跳过**（不报错也
    不告警），于是「LoRA 看起来加载成功但效果为 0」这类问题极难定位。
    这里按 LoRA 的 target 名去 transformer 的模块名里查，给运维一个显式
    信号（对齐 ComfyUI load_lora_for_models 的 `NOT LOADED xxx` 告警）。
    """
    import re

    tr = getattr(pipe, "transformer", None)
    if tr is None:
        return []
    try:
        from safetensors import safe_open
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
    except Exception:  # noqa: BLE001
        return []

    tgt = {name for name, _ in tr.named_modules()}
    bad = set()
    for k in keys:
        m = re.match(r"^(.+?)\.(lora_A|lora_B|lora_down|lora_up)\b", k)
        if not m:
            continue
        name = m.group(1)
        cand = [name]
        if name.startswith("transformer."):
            cand.append(name[len("transformer."):])
        if not any(c in tgt for c in cand):
            bad.add(name)
    return sorted(bad)


def _load_lora(pipe, spec, scale, into_transformer_2=False):
    """逗号分隔多挂 LoRA（可带 @strength），Wan 版。

    与 H3 的差异：Wan2.2 有两个去噪器，默认只挂第一个；into_transformer_2
    时同时挂到低噪专家（LightX2V 等加速 LoRA 需要两边都有才有效）。
    """
    if not spec:
        return pipe

    names = [s.strip() for s in spec.split(",") if s.strip()]
    for idx, item in enumerate(names):
        if "@" in item:
            name, s = item.rsplit("@", 1)
            strength = float(s)
        else:
            name, strength = item, float(scale)
        path = _resolve("loras", name.strip())
        _require_file(path, f"LoRA {name}")
        adapter = f"wan_lora_{idx}"
        kwargs = {"adapter_name": adapter}
        if into_transformer_2:
            kwargs["load_into_transformer_2"] = True
        try:
            pipe.load_lora_weights(path, **kwargs)
        except TypeError:
            kwargs.pop("load_into_transformer_2", None)
            pipe.load_lora_weights(path, **kwargs)
        pipe.set_adapters(adapter, adapter_weights=strength)
        # **未匹配 key 点名**（对齐 ComfyUI `NOT LOADED xxx`）：
        # diffusers 的 load_lora_weights 遇到不认识的 key 只会静默忽略，
        # 表现为「LoRA 挂上了、流程 success、效果没变」——最难排查的一类。
        unmatched = _lora_unmatched(pipe, path)
        if unmatched:
            print(f"[wan] ⚠ LoRA '{os.path.basename(path)}' 有 "
                  f"{len(unmatched)} 个 key 未匹配（样例 {unmatched[:5]}）",
                  flush=True)
        print(f"[wan] LoRA '{os.path.basename(path)}' "
              f"strength={strength} (transformer_2={into_transformer_2})",
              flush=True)
    return pipe


# ---------------------------------------------------------------------------
# text encoder / vae / image encoder
# ---------------------------------------------------------------------------

def _build_text_encoder(path, dtype):
    """UMT5-XXL 单文件 → transformers UMT5EncoderModel。

    Comfy-Org 的 umt5_xxl_fp8_e4m3fn_scaled.safetensors 是 fp8 标量量化，
    官方给了 fp8_scaled 的 weight_scale；mxfp8/fp8_e4m3fn 需按标量反量化。
    本项目先用 bf16 单文件（共享盘通常有 umt5_xxl_bf16）：
    14B 全链路 CPU 常驻 ~11GB 在 64GB 内存预算内，无需 int8。
    """
    import torch
    from safetensors.torch import load_file
    from transformers import UMT5EncoderModel

    t0 = time.time()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    if os.path.isdir(path):
        # HF/ModelScope 的 text_encoder 目录（config.json + 分片）
        from transformers import UMT5EncoderModel
        model = UMT5EncoderModel.from_pretrained(path, torch_dtype=torch_dtype)
        print(f"[wan] text_encoder from_pretrained {time.time()-t0:.1f}s",
              flush=True)
        return model
    cfg = _config_json("text_encoder") or {
        "_name_or_path": "google/umt5-xxl",
        "architectures": ["UMT5EncoderModel"],
        "d_model": 4096,
        "d_ff": 10240,
        "d_kv": 64,
        "num_layers": 24,
        "num_heads": 64,
        "relative_attention_num_buckets": 32,
        "model_type": "umt5",
        "vocab_size": 256384,
        "feed_forward_proj": "gated-gelu",
        "is_encoder_decoder": False,
        "use_cache": False,
        "tokenizer_class": "T5TokenizerFast",
    }
    sd = load_file(path)
    # Comfy-Org 单文件 key 常带 encoder. / model. 前缀，逐层剥
    for prefix in ("encoder.", "model.", "text_encoder."):
        if any(k.startswith(prefix) for k in sd) and not any(
                k.startswith("encoder.") for k in sd):
            sd = {k[len(prefix):] if k.startswith(prefix) else k: v
                  for k, v in sd.items()}
    if "shared.weight" not in sd:
        for k in list(sd)[:5]:
            print(f"[wan] TE key 样例: {k}", flush=True)
    from transformers import UMT5Config
    hf_cfg = UMT5Config(**{k: v for k, v in cfg.items()
                           if k in UMT5Config().to_dict()})
    model = UMT5EncoderModel(hf_cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[wan] TE missing {len(missing)} keys "
              f"（样例 {missing[:3]}）", flush=True)
    if unexpected:
        print(f"[wan] TE unexpected {len(unexpected)} keys "
              f"（样例 {unexpected[:3]}）", flush=True)
    model = model.to(torch_dtype)
    print(f"[wan] text_encoder ready {time.time()-t0:.1f}s", flush=True)
    return model


def _build_vae(path):
    """Wan VAE → AutoencoderKLWan，**强制 fp32**（官方：解码质量更好）。"""
    import torch
    from diffusers import AutoencoderKLWan

    t0 = time.time()
    if os.path.isdir(path):
        vae = AutoencoderKLWan.from_pretrained(path, torch_dtype=torch.float32)
        print(f"[wan] vae from_pretrained fp32 {time.time()-t0:.1f}s", flush=True)
        return vae
    # 同 transformer：config 必须是路径（dict 会被 diffusers 拒绝）
    cfg_dir = os.path.join(WAN_CONFIG_DIR, "vae")
    if os.path.isfile(os.path.join(cfg_dir, "config.json")):
        vae = AutoencoderKLWan.from_single_file(
            path, config=cfg_dir, torch_dtype=torch.float32)
    else:
        print("[wan] vae 无 config 目录，按内置结构兜底", flush=True)
        from safetensors.torch import load_file
        vae = AutoencoderKLWan.from_config(dict(_VAE_CFG))
        sd = load_file(path)
        for pre in ("first_stage_model.", "vae.", "model."):
            if all(k.startswith(pre) for k in sd):
                sd = {k[len(pre):]: v for k, v in sd.items()}
                break
        missing, unexpected = vae.load_state_dict(sd, strict=False)
        if missing:
            print(f"[wan] ⚠ vae missing {len(missing)} keys"
                  f"（样例 {missing[:3]}）", flush=True)
    vae = vae.to(torch.float32)
    print(f"[wan] vae ready fp32 {time.time()-t0:.1f}s", flush=True)
    return vae


def _build_image_encoder(path):
    """CLIP-ViT-H（I2V / Animate 的图像条件编码器），fp32。"""
    import torch
    from transformers import CLIPVisionModel

    cfg = _config_json("image_encoder")
    if cfg is None:
        cfg = {"_name_or_path": "openai/clip-vit-huge-patch14",
               "architectures": ["CLIPVisionModel"],
               "hidden_size": 1280, "image_size": 224,
               "intermediate_size": 5120, "model_type": "clip_vision_model",
               "num_attention_heads": 16, "num_hidden_layers": 32,
               "patch_size": 14, "projection_dim": 1024}
    from transformers import CLIPVisionConfig
    hf_cfg = CLIPVisionConfig(
        **{k: v for k, v in cfg.items()
           if k in CLIPVisionConfig().to_dict()})
    m = CLIPVisionModel(hf_cfg)
    from safetensors.torch import load_file
    sd = load_file(path)
    # **逐键剥离（2026-11-14 真机 519 missing 复盘）**：原先用
    # `all(k.startswith(prefix))` 判断整体前缀，但 clip_vision_h.safetensors
    # 里混着 `visual_projection.weight` / `logit_scale` 两个**不带前缀**的
    # 顶层键 → all() 失败 → 前缀没剥 → 519 键全 missing → CLIP 等于随机
    # 初始化 → image_embeds 垃圾 → 采样第一步就 NaN。
    # 正确做法：带前缀的逐键剥，CLIPVisionModel 不吃的顶层键丢弃。
    sd = {(k[len("vision_model."):] if k.startswith("vision_model.") else k): v
          for k, v in sd.items()}
    sd = {k: v for k, v in sd.items()
          if not k.startswith(("visual_projection", "logit_scale",
                               "text_model")) and k != "position_ids"}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[wan] image_encoder missing {len(missing)} keys"
              f"（样例 {missing[:3]}）", flush=True)
    if unexpected:
        print(f"[wan] image_encoder unexpected {len(unexpected)} keys"
              f"（样例 {unexpected[:3]}）", flush=True)
    return m.to(torch.float32)


# 可选调度器（对齐 ComfyUI 的「采样器 × 调度器」二维选择思路：模型族给出
# 合法集，用户按画质/速度取舍）。Wan 是 flow-matching 模型，ddim/euler_a 这类
# 非 flow 调度器不适用，故只给 flow 兼容集：
_SCHEDULERS = {
    # UniPC 多步：**需配套 Wan2.1 自己的 scheduler config**，套 Wan2.2 的会出噪点
    "unipc": "UniPCMultistepScheduler",
    # 一阶 Euler（flow match）：**真机验证可用的默认**，支持 flow_shift
    "euler": "FlowMatchEulerDiscreteScheduler",
    # DPM++ 2M（flow match）：中低步数画质好，支持 flow_shift
    "dpm++": "FlowMatchDPMSolverMultistepScheduler",
    # 二阶 Heun：质量略高于 Euler，步数翻倍（慢）
    "heun": "FlowMatchHeunDiscreteScheduler",
}


def _is_vace_pipe(pipe):
    """判断 pipe 是否为 VACE（按类名 + transformer 结构双重判断）。"""
    try:
        if "vace" in type(pipe).__name__.lower():
            return True
    except Exception:  # noqa: BLE001
        pass
    tr = getattr(pipe, "transformer", None)
    cfg = getattr(tr, "config", None)
    if cfg is not None and getattr(cfg, "vace_in_channels", None):
        return True
    return False


    @registry.register(
        "wan.vace_standalone",
        inputs={"transformer": "TRANSFORMER", "text_encoder": "TEXT_ENCODER",
                "vae": "VAE", "video": "VIDEO", "mask": "VIDEO",
                "reference_image": "IMAGE", "prompt": "STRING",
                "negative_prompt": "STRING", "steps": "INT",
                "guidance_scale": "FLOAT", "flow_shift": "FLOAT",
                "scheduler": "STRING", "conditioning_scale": "FLOAT",
                "height": "INT", "width": "INT", "num_frames": "INT",
                "seed": "INT"},
        outputs={"video": "VIDEO", "seed": "INT", "fps": "INT"},
        description="Wan VACE 视频重绘/人物替换（**独立装配**）。直接接 "
                    "transformer / text_encoder / vae 三个组件句柄，内部按"
                    "diffusers 官方示例 `WanVACEPipeline(...) + pipe.to(cuda)` "
                    "装配，**不经过 wan.pipe**（绕开组件缓存与 offload hook）。")
    def _op_vace_standalone(transformer, text_encoder, vae, video=None,
                            mask=None, reference_image=None, prompt="",
                            negative_prompt="", steps=20,
                            guidance_scale=5.0, flow_shift=5.0,
                            scheduler="euler", conditioning_scale=1.0,
                            height=0, width=0, num_frames=0, seed=0):
        import torch
        from diffusers import WanVACEPipeline
        from transformers import AutoTokenizer

        if transformer is None or text_encoder is None or vae is None:
            raise ValueError("wan.vace_standalone: transformer / "
                             "text_encoder / vae 三者都必填")
        tok_src = os.path.join(WAN_CONFIG_DIR, "tokenizer")
        if not os.path.isdir(tok_src):
            tok_src = os.path.join(WAN_CONFIG_DIR, "tokenizer_2")
        if not os.path.isdir(tok_src):
            raise ValueError(f"wan: 缺 tokenizer 目录（{tok_src}）")
        tokenizer = AutoTokenizer.from_pretrained(tok_src)
        sch = _build_scheduler(flow_shift, scheduler)

        print("[wan] vace_standalone 独立装配（官方路径）", flush=True)
        pipe = WanVACEPipeline(tokenizer=tokenizer,
                               text_encoder=text_encoder, vae=vae,
                               transformer=transformer, scheduler=sch)
        # 官方示例就是 pipe.to("cuda")。1.3B(6.7G) + UMT5(11G) ≈ 18G 放得下；
        # 放不下（14B/32G）时交给 group offloading 兜底。
        try:
            pipe.to("cuda")
            print("[wan] vace_standalone 全量上卡完成", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[wan] vace_standalone 全量上卡失败"
                  f"({type(e).__name__}) → group offloading", flush=True)
            try:
                from diffusers.hooks import apply_group_offloading
                for _m in (transformer, text_encoder):
                    apply_group_offloading(
                        _m, onload_device=torch.device("cuda"),
                        offload_device=torch.device("cpu"),
                        offload_type="block_level",
                        num_blocks_per_group=1, use_stream=True,
                        low_cpu_mem_usage=True)
                vae.to("cuda")
            except Exception as e2:  # noqa: BLE001
                raise RuntimeError(
                    f"wan.vace_standalone 显存不足：{type(e2).__name__}: {e2}"
                ) from e2
        pipe._flowx_clean = True
        return _op_vace(pipe, prompt=prompt,
                        negative_prompt=negative_prompt, video=video,
                        mask=mask, reference_image=reference_image,
                        conditioning_scale=conditioning_scale,
                        guidance_scale=guidance_scale, steps=steps,
                        height=height, width=width, num_frames=num_frames,
                        seed=seed)


def _build_scheduler(flow_shift, name="euler"):
    """按名字构建 flow-matching 调度器（**默认 euler**）。

    ⚠ 真机教训（2026-11-11）：默认用 unipc + 共享盘 Wan2.2 的 scheduler config
    会产出**抽象色块**（`use_flow_sigmas/time_shift_type=exponential` 与
    Wan2.1 flow-matching 不匹配）。改成 FlowMatchEulerDiscrete + flow_shift
    后立即出正常画面。

    flow_shift：720p→5.0 / 480p→3.0（官方建议）；UniPC 与 DPM++ 都吃这个参数。
    """
    import diffusers

    key = str(name or "unipc").lower().strip()
    if key not in _SCHEDULERS:
        raise ValueError(
            f"wan: 未知 scheduler '{name}'，可选 {sorted(_SCHEDULERS)}")
    cls_name = _SCHEDULERS[key]
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        # 老版本 diffusers 缺该调度器 → 明确回退而不是静默换
        print(f"[wan] diffusers 无 {cls_name}，回退 unipc", flush=True)
        cls = diffusers.UniPCMultistepScheduler
        key = "unipc"

    # **euler 走干净构造（2026-11-16 雪花事故真凶）**：Wan2.2-T2V 的
    # scheduler_config.json 里 shift=3.0 / use_flow_sigmas / time_shift_type
    # =exponential 等字段与 Wan-Animate 的 flow-matching 不兼容，读进来会把
    # Euler 的 sigma 表改坏 → transformer 每步朝随机方向去噪 → 成片人物区
    # 一片彩色雪花（背景是原帧混回，完好）。官方 WanAnimatePipeline 示例
    # 只用 `FlowMatchEulerDiscreteScheduler(shift=5.0)`，真机 3090 实测正常。
    # 故 euler 不读外部 config，仅保留 flow_shift + 训练步数。
    if key == "euler":
        import diffusers as _df
        _sched = _df.FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000, shift=float(flow_shift))
        print(f"[wan] scheduler=euler(clean) shift={float(flow_shift)} "
              f"（不读 Wan2.2 scheduler config）", flush=True)
        return _sched
    cfg = _config_json("scheduler") or {}
    # **flow-matching 兜底（2026-11-14）**：Wan 全系必须 flow_prediction +
    # flow sigmas；config 缺失时绝不能退回 epsilon 默认（出噪声的元凶）。
    cfg.setdefault("prediction_type", "flow_prediction")
    cfg.setdefault("use_flow_sigmas", True)
    cfg.setdefault("solver_order", 2)
    cfg.setdefault("solver_type", "bh2")
    cfg.setdefault("lower_order_final", True)
    cfg.update({"num_train_timesteps": 1000})
    # flow_shift 只有部分调度器接受，按签名探测后传（不支持的忽略并提示）
    import inspect
    params = set(inspect.signature(cls.__init__).parameters)
    if "flow_shift" in params:
        cfg["flow_shift"] = float(flow_shift)
    elif float(flow_shift) != 5.0:
        print(f"[wan] {key} 不支持 flow_shift，忽略（仅 UniPC/DPM++ 支持）",
              flush=True)
    print(f"[wan] scheduler={key} ({cls_name}) "
          f"flow_shift={cfg.get('flow_shift', 'n/a')}", flush=True)
    return cls.from_config(cfg)


def _align_frames(n):
    """Wan 帧数契约 **4k+1**（81/121/161…）。"""
    n = max(5, int(n))
    k = max(1, round((n - 1) / 4))
    return 4 * k + 1


# ---------------------------------------------------------------------------
# 算子注册
# ---------------------------------------------------------------------------

def _assemble_pipe(task, tr_info, te_info, vae_info, img_enc_info,
                   scheduler, boundary_ratio, extra=None):
    """三管道装配（一体版与拆分版共用）。

    tr_info   {"model": transformer, "model2": transformer_2|None, "is_moe": b}
    te_info   {"model": text_encoder, "tokenizer": tokenizer}
    vae_info  {"vae": autoencoder_kl_wan}
    img_enc_info {"model": clip_vision|None}
    """
    import torch
    from diffusers import (WanAnimatePipeline, WanImageToVideoPipeline,
                           WanPipeline, WanVACEPipeline)

    common = dict(tokenizer=te_info["tokenizer"], text_encoder=te_info["model"],
                  vae=vae_info["vae"], scheduler=scheduler,
                  transformer=tr_info["model"])
    if tr_info.get("is_moe") and tr_info.get("model2") is not None:
        common["transformer_2"] = tr_info["model2"]
        common["boundary_ratio"] = float(boundary_ratio)

    if task == "animate2":
        # **2026-11-14 路线纠错**：弃用 WanAnimate2ModularPipeline（新架构，
        # 与 Wan2.2-Animate-14B 权重不匹配，丢 138 键 + ModularPipeline 内部
        # 22GB 不明显存占用）。改用 WanAnimatePipeline（普通 pipeline，
        # mode="animate"/"replace" 双模式，真机已端到端出片验证）。
        # image_processor 从官方 Diffusers 仓库小件加载。
        from transformers import CLIPImageProcessor
        _ip_dir = os.environ.get(
            "WAN_ANIMATE_IP", "/models/wan-animate-diffusers/image_processor")
        image_processor = CLIPImageProcessor.from_pretrained(_ip_dir)
        # **Relighting LoRA（2026-11-15 官方配方对齐）**：replace 人物光照
        # 匹配背景（治"人物像另一个图层"）。extra.relight_lora 给路径即启用；
        # merge 进权重（挂 offload hook 之前做），组件幂等。
        _relight = str((extra or {}).get("relight_lora", "") or "")
        if _relight:
            _merge_relight_lora(tr_info["model"], _relight)
        pipe = WanAnimatePipeline(
            tokenizer=te_info["tokenizer"], text_encoder=te_info["model"],
            vae=vae_info["vae"], scheduler=scheduler,
            transformer=tr_info["model"],
            image_encoder=img_enc_info.get("model"),
            image_processor=image_processor)
        # **显存（2026-11-13/14 真机定案）**：bf16 transformer ≈34GB > 24GB，
        # cpu_offload 整件换入必 OOM；apply_group_offloading 块级流式是唯一
        # 正解（实测峰值 17.5GB）。text_encoder/image_encoder 同挂，vae 常驻。
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()
        # **装配幂等（2026-11-15 真机死锁复盘）**：组件（transformer/TE/CLIP）
        # 在 _MODELS 缓存里跨执行复用，而 pipeline 每次执行重新装配——
        # apply_group_offloading 在同一组件上**重复挂载会死锁**
        # （GPU/CPU 双 0% 互等 stream event，1435 第三次装配卡死实证）。
        # 组件挂了标记位就跳过重挂与预热，pipeline 只管组装。
        _already = bool(getattr(tr_info["model"], "_flowx_offload_armed", False))
        try:
            from diffusers.hooks import apply_group_offloading
            # ⚠ transformer blocks_per_group 必须=1（2026-11-14 真机）：
            # =2 + use_stream 在 WanAnimateTransformer3DModel 上死锁
            # （GPU/CPU 双双 0% 互等 stream event）。=1 是 1412/1413/1416
            # 验证过的安全组合（23s/步）。
            # **TE/CLIP 常驻 cuda 与否（2026-11-16 雪花排查）**：官方
            # WanAnimatePipeline 示例把 text_encoder/image_encoder 直接
            # .to("cuda") 常驻、只对 transformer 做 block offload。我们原先
            # 把三个组件都挂 group offloading，若某架构上搬运未正确同步，
            # prompt/参考图嵌入会被污染 → 采样全程朝随机方向 → 雪花。
            # WAN_TE_ON_CUDA=0 可回到"全部 offload"的旧行为。
            _te_cuda = os.environ.get("WAN_TE_ON_CUDA", "0") == "1"
            _pairs = ([("transformer", 1)] if _te_cuda else
                      [("transformer", 1), ("text_encoder", 2),
                       ("image_encoder", 1)])
            if _te_cuda:
                for _cn in ("text_encoder", "image_encoder"):
                    _cc = getattr(pipe, _cn, None)
                    if _cc is not None:
                        _cc.to(torch.device("cuda"))
                print("[wan] TE/CLIP 常驻 cuda（对齐官方示例）", flush=True)
            for _cname, _npg in _pairs:
                if _already:
                    break
                _c = getattr(pipe, _cname, None)
                if _c is None:
                    continue
                apply_group_offloading(
                    _c, onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    offload_type="block_level", num_blocks_per_group=_npg,
                    # use_stream=True：异步搬运与计算重叠（快 ~2.5x）。
                    # 曾怀疑它导致间歇 NaN（2026-11-14）——后续查明真凶是
                    # fp16 T5 溢出 + CLIP 519 missing + scheduler 错配置，
                    # 与 stream 无关，恢复开启（blocks_per_group=2 再提速）。
                    # **use_stream 环境开关（2026-11-16）**：4090 上开
                    # stream 正常；某些架构组合下异步搬运与计算可能未正确
                    # 同步 → 前向读到未到位权重 → 输出全随机噪点（3090 实证）。
                    # WAN_OFFLOAD_STREAM=0 关闭，退化为同步搬运（稍慢但确定）。
                    use_stream=(os.environ.get("WAN_OFFLOAD_STREAM", "1") != "0"),
                    low_cpu_mem_usage=True)
                print(f"[wan] {_cname}: group_offloading(block={_npg})",
                      flush=True)
            if not _already:
                pipe.transformer._flowx_offload_armed = True
            pipe.vae.to("cuda")
            print(f"[wan] vae -> cuda (offload {'复用已挂' if _already else '新挂'})",
                  flush=True)
            # **装配期预热（2026-11-15）**：group_offloading 的首次 block 搬运 /
            # pinned buffer 分配 / CUDA context / T5 流转全发生在首个正式
            # 前向（实测首步 9~20 分钟）。这里用 64×64×5帧×1步 dummy 把全部
            # 路径热一遍（~2 分钟），Animate 首步即热速度。失败不影响正式跑。
            if _already:
                print("[wan] 组件复用，跳过装配期预热", flush=True)
            # **预热可关（2026-11-16 雪花排查）**：官方 WanAnimatePipeline
            # 示例没有"装配期 dummy 前向"这一环。预热会在 offload hook 已
            # 挂载的状态下跑一次前向，若某些架构上这次前向留下 KV/rope 缓存
            # 或搬运状态未复位，正式生成就会全程走偏。WAN_WARMUP=0 跳过。
            elif (not extra or extra.get("warmup", True)) and \
                    os.environ.get("WAN_WARMUP", "1") != "0":
                try:
                    from PIL import Image as _WI
                    t0 = time.time()
                    black = _WI.new("RGB", (64, 64), (0, 0, 0))
                    pipe(image=black, pose_video=[black] * 5,
                         face_video=[black] * 5, prompt="warmup",
                         height=64, width=64, num_inference_steps=1,
                         guidance_scale=1.0, mode="animate",
                         output_type="np")
                    torch.cuda.synchronize()
                    print(f"[wan] 装配期预热完成 {time.time()-t0:.1f}s "
                          f"(Animate 首步将直接全速)", flush=True)
                except Exception as _we:  # noqa: BLE001
                    print(f"[wan] ⚠ 预热失败（不影响正式跑）: "
                          f"{type(_we).__name__} {str(_we)[:100]}", flush=True)
        except Exception as _e:  # noqa: BLE001
            print(f"[wan] ⚠ group_offloading 失败（{type(_e).__name__}: "
                  f"{str(_e)[:120]}）→ 退回 enable_model_cpu_offload", flush=True)
            pipe.enable_model_cpu_offload()
    elif task == "vace":
        pipe = WanVACEPipeline(**common)
    elif task == "animate":
        pipe = WanAnimatePipeline(**common)
    elif task == "i2v":
        pipe = WanImageToVideoPipeline(
            image_encoder=img_enc_info.get("model"), **common)
    else:
        pipe = WanPipeline(**common)

    # 可选：VAE tiling/slicing（大分辨率解码省显存）
    if extra:
        if extra.get("vae_tiling"):
            try:
                pipe.vae.enable_tiling()
                print("[wan] vae tiling on", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[wan] vae tiling 不可用: {type(e).__name__}", flush=True)
        if extra.get("vae_slicing"):
            try:
                pipe.vae.enable_slicing()
                print("[wan] vae slicing on", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[wan] vae slicing 不可用: {type(e).__name__}", flush=True)
        if extra.get("attention") in ("sdpa", "flash_attention_2"):
            mode = extra["attention"]
            for attr in ("transformer", "transformer_2"):
                m = getattr(pipe, attr, None)
                if m is None:
                    continue
                try:
                    m.set_attn_processor(m.attn_processors)  # noqa: PLW2901
                except Exception:  # noqa: BLE001
                    pass
            try:
                pipe.transformer.set_attention_backend(mode)
                if getattr(pipe, "transformer_2", None) is not None:
                    pipe.transformer_2.set_attention_backend(mode)
                print(f"[wan] attention backend -> {mode}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[wan] attention backend {mode} 不可用"
                      f"（{type(e).__name__}），用默认", flush=True)
    return pipe


def _build_transformer_dispatch(path, name, dtype, which="transformer"):
    """按权重名分派 transformer 构建器（一体版与拆分版共用，防两份实现漂移）。

    为何必须分派（真机：拆分版 `wan.transformer_load(transformer="wan-vace-1.3b")`
    报 `Invalid pretrained_model_name_or_path`）：
      - VACE：in_channels=96 + added_kv_proj_dim，**普通 WanTransformer3DModel
        装不下**，必须用 WanVACETransformer3DModel
      - Animate-2：原始 WanAnimateModel 格式，要过官方转换函数
    一体版原先在 _load_components 里内联判断，拆分版漏了 → 补成共用 helper。
    """
    low = (name or "").lower()
    is_animate2 = "animate" in low and (
        "animate-2" in low or "animate2" in low or "animate_2" in low)
    if is_animate2:
        print("[wan] 检测到 Animate-2 权重 → Animate-2 装配路径", flush=True)
        return _build_animate2_transformer(path, dtype, which)
    if "vace" in low:
        print("[wan] 检测到 VACE transformer → VACE 装配路径", flush=True)
        return _build_vace_transformer(path, dtype, which)
    return _build_transformer(path, dtype, which)


def _load_components(transformer, transformer2, text_encoder, vae,
                     image_encoder, dtype, offload, key_prefix="wan:load"):
    """加载全部组件（一体版与拆分版共用），返回 (tr_info, te_info,
    vae_info, img_info, cache_key, weight_gb)。"""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_path = _resolve("diffusion_models", transformer)
    _require_file(tr_path, "transformer")
    tr = _build_transformer_dispatch(tr_path, transformer, dtype, "transformer")
    tr = _offload(tr, device, offload, "transformer")
    tr2 = None
    if transformer2:
        tr2_path = _resolve("diffusion_models", transformer2)
        _require_file(tr2_path, "transformer2")
        tr2 = _build_transformer_dispatch(tr2_path, transformer2, dtype,
                                          "transformer_2")
        tr2 = _offload(tr2, device, offload, "transformer_2")

    te_path = _resolve("text_encoders", text_encoder)
    _require_file(te_path, "text_encoder")
    te = _build_text_encoder(te_path, dtype)
    te = _offload(te, device, offload, "text_encoder")

    tok_src = os.path.join(WAN_CONFIG_DIR, "tokenizer")
    if not os.path.isdir(tok_src):
        tok_src = os.path.join(WAN_CONFIG_DIR, "tokenizer_2")
    if not os.path.isdir(tok_src):
        raise ValueError(f"wan: 缺 tokenizer 目录（{WAN_CONFIG_DIR}/tokenizer）"
                         f"——umt5 词表须从官方 Diffusers 仓库 copy，KB 级")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tok_src)

    vae_path = _resolve("vae", vae)
    _require_file(vae_path, "vae")
    vae_m = _build_vae(vae_path)
    if device.type == "cuda":
        vae_m = vae_m.to(device)

    img_info = {"model": None}
    if image_encoder:
        img_enc = _build_image_encoder(
            _resolve("image_encoders", image_encoder))
        if device.type == "cuda":
            img_enc = img_enc.to(device)
        img_info = {"model": img_enc}

    gb = _size(tr_path) + (_size(_resolve("diffusion_models", transformer2))
                           if transformer2 else 0)
    return ({"model": tr, "model2": tr2, "is_moe": bool(transformer2)},
            {"model": te, "tokenizer": tokenizer},
            {"vae": vae_m}, img_info, gb)


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)


    # ============ 拆分装配算子（SD 风：组件各自加载/设置，pipe 组装）============
    # 缓存键带组件前缀（wan:tr:/te:/vae:/img:/pipe:），与一体版 wan:<task>: 互不
    # 冲突。组件常驻 model_manager LRU（跨执行复用：换 VAE 不重装 28GB
    # transformer）。pipe est=0——组件已各自记账，不重复计。

    @registry.register(
        "wan.transformer_load",
        inputs={"transformer": "STRING", "transformer2": "STRING",
                "dtype": "STRING", "offload": "STRING"},
        outputs={"transformer": "TRANSFORMER"},
        description="Wan transformer 组件加载（+ Wan2.2 MoE 低噪专家、"
                    "可选 LoRA 双挂）。14B bf16 ~28GB 走逐层 offload。")
    def _op_transformer_load(
            transformer="wan2.1_t2v_14B_bf16.safetensors",
            transformer2="", dtype="bf16", offload="sequential"):
        import torch

        key = f"wan:tr:{transformer}|{transformer2}|{dtype}|{offload}"
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"transformer": models._pipes[key]}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr_path = _resolve("diffusion_models", transformer)
        _require_file(tr_path, "transformer")
        tr = _build_transformer_dispatch(tr_path, transformer, dtype,
                                         "transformer")
        tr = _offload(tr, device, offload, "transformer")
        tr2 = None
        if transformer2:
            tr2_path = _resolve("diffusion_models", transformer2)
            _require_file(tr2_path, "transformer2")
            tr2 = _build_transformer_dispatch(tr2_path, transformer2, dtype,
                                              "transformer_2")
            tr2 = _offload(tr2, device, offload, "transformer_2")

        info = {"model": tr, "model2": tr2, "is_moe": bool(transformer2)}
        est = _size(tr_path) + (_size(_resolve("diffusion_models",
                                               transformer2))
                                if transformer2 else 0)
        # **账本口径 vs 实际行为（2026-11-13 真机）**：`offload="none"` 是给
        # `apply_group_offloading`（在 wan.animate2 里按 block 流式装载）用的，
        # 但显存闸门只看 offload_mode —— 传 none 会按「权重全量常驻」估算
        # （Animate2 14B 算出 86GB）直接拒绝加载。故账本仍按 sequential 记。
        _budget_mode = "sequential" if str(offload).lower() in ("none", "") \
            else offload
        return {"transformer": _register(key, info, est,
                                         "WanTransformer3DModel",
                                         offload_mode=_budget_mode)}

    @registry.register(
        "wan.te_load",
        inputs={"text_encoder": "STRING", "dtype": "STRING",
                "offload": "STRING"},
        outputs={"text_encoder": "TEXT_ENCODER"},
        description="Wan 文本编码器（UMT5-XXL，11GB bf16）组件加载 + "
                    "tokenizer（从 WAN_DIR/config/tokenizer 读 umt5 词表）。")
    def _op_te_load(text_encoder="umt5_xxl_bf16.safetensors",
                    dtype="bf16", offload="sequential"):
        import torch

        key = f"wan:te:{text_encoder}|{dtype}|{offload}"
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"text_encoder": models._pipes[key]}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        te_path = _resolve("text_encoders", text_encoder)
        _require_file(te_path, "text_encoder")
        te = _build_text_encoder(te_path, dtype)
        te = _offload(te, device, offload, "text_encoder")

        tok_src = os.path.join(WAN_CONFIG_DIR, "tokenizer")
        if not os.path.isdir(tok_src):
            tok_src = os.path.join(WAN_CONFIG_DIR, "tokenizer_2")
        if not os.path.isdir(tok_src):
            raise ValueError(
                f"wan: 缺 tokenizer 目录（{WAN_CONFIG_DIR}/tokenizer）")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_src)

        info = {"model": te, "tokenizer": tokenizer}
        return {"text_encoder": _register(key, info, _size(te_path),
                                          "UMT5EncoderModel",
                                          offload_mode=offload)}

    @registry.register(
        "wan.vae_load",
        inputs={"vae": "STRING", "device": "STRING"},
        outputs={"vae": "VAE"},
        description="Wan VAE 加载（**强制 fp32**——官方：fp32 解码质量更好；"
                    "体积小 ~500MB 不构成显存压力）。")
    def _op_vae_load(vae="wan_2.1_vae.safetensors", device="cuda"):
        import torch

        key = f"wan:vae:{vae}"
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"vae": models._pipes[key]}

        vae_path = _resolve("vae", vae)
        _require_file(vae_path, "vae")
        vae_m = _build_vae(vae_path)
        if device != "cpu" and torch.cuda.is_available():
            vae_m = vae_m.to("cuda")
        return {"vae": _register(key, {"vae": vae_m}, _size(vae_path),
                                 "AutoencoderKLWan")}

    @registry.register(
        "wan.image_encoder_load",
        inputs={"image_encoder": "STRING", "device": "STRING"},
        outputs={"image_encoder": "CLIP_VISION"},
        description="Wan 图像编码器（CLIP-ViT-H，I2V/Animate 的图像条件）加载。")
    def _op_image_encoder_load(
            image_encoder="clip_vision_h.safetensors", device="cuda"):
        import torch

        key = f"wan:img:{image_encoder}"
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"image_encoder": models._pipes[key]}

        path = _resolve("image_encoders", image_encoder)
        _require_file(path, "image_encoder")
        m = _build_image_encoder(path)
        if device != "cpu" and torch.cuda.is_available():
            m = m.to("cuda")
        return {"image_encoder": _register(key, {"model": m}, _size(path),
                                           "CLIPVisionModel")}

    @registry.register(
        "wan.pipe",
        inputs={"task": "STRING", "transformer": "TRANSFORMER",
                "text_encoder": "TEXT_ENCODER", "vae": "VAE",
                "image_encoder": "CLIP_VISION",
                "boundary_ratio": "FLOAT", "flow_shift": "FLOAT",
                "lora": "STRING", "lora_scale": "FLOAT",
                "vae_tiling": "BOOL", "vae_slicing": "BOOL",
                "attention": "STRING", "max_sequence_length": "INT",
                "scheduler": "STRING", "relight_lora": "STRING"},
        outputs={"model": "MODEL"},
        description="Wan 管道组装：把 transformer/text_encoder/vae/"
                    "image_encoder 组件装成可生成管道（task=t2v/i2v/animate）。"
                    "换 VAE 或采样参数不必重装 28GB transformer。")
    def _op_pipe(task="t2v", transformer=None, text_encoder=None, vae=None,
                 image_encoder=None, boundary_ratio=0.875, flow_shift=5.0,
                 scheduler="euler", lora="", lora_scale=1.0,
                 vae_tiling=False, vae_slicing=False, attention="default",
                 max_sequence_length=512, relight_lora=""):
        # **独立装配直通（2026-11-16）**：WAN_STANDALONE=1 时 wan.animate2
        # 会自己 from_pretrained 全套组件（照抄官方示例），此时 wan.pipe
        # 再装一份就是双份 33GB transformer → 必然 OOM。这里直接返回占位，
        # 只保留结构（下游只把 model 当句柄透传，真正装配在 animate2 内部）。
        if os.environ.get("WAN_STANDALONE", "1") == "1" and \
                (task or "").lower() == "animate2":
            print("[wan] pipe: WAN_STANDALONE=1 → 跳过装配（animate2 自装配）",
                  flush=True)
            return {"model": {"__standalone__": True, "task": "animate2"}}
        if transformer is None or text_encoder is None or vae is None:
            raise ValueError(
                "wan.pipe: transformer / text_encoder / vae 三个组件必填")
        task = (task or "t2v").lower()
        if task not in ("t2v", "i2v", "animate", "animate2", "vace"):
            raise ValueError(
                f"wan.pipe: task 须为 t2v/i2v/animate/animate2/vace，得 {task}")

        tr_info = transformer
        is_moe = bool(tr_info.get("is_moe"))
        key = (f"wan:pipe:{task}:{id(tr_info.get('model')):x}|"
               f"{id(tr_info.get('model2')) if is_moe else 0:x}|"
               f"{id(text_encoder.get('model')):x}|"
               f"{id(vae.get('vae')):x}|"
               f"{id((image_encoder or {}).get('model')):x}|"
               f"{boundary_ratio}|{flow_shift}|{vae_tiling}|{vae_slicing}|"
               f"{attention}|{max_sequence_length}|{scheduler}")
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"model": models._pipes[key]}

        sched = _build_scheduler(flow_shift, scheduler)
        pipe = _assemble_pipe(
            task, tr_info, text_encoder, vae, image_encoder or {}, sched,
            boundary_ratio,
            extra={"vae_tiling": vae_tiling, "vae_slicing": vae_slicing,
                   "attention": attention,
                   "relight_lora": str(relight_lora or "")})
        if lora:
            pipe = _load_lora(pipe, lora, lora_scale, into_transformer_2=is_moe)
        pipe._wan_fps = int(getattr(pipe, "_wan_fps", WAN_FPS_DEFAULT))
        pipe._wan_max_seq = int(max_sequence_length)
        print(f"[wan] pipe assembled from components ({task}"
              f"{', MoE' if is_moe else ''})", flush=True)
        return {"model": _register(
            key, pipe, 0,
            f"Wan{'Animate' if task == 'animate' else 'Pipeline'}")}

    @registry.register(
        "wan.encode",
        inputs={"model": "MODEL", "prompt": "STRING",
                "negative_prompt": "STRING",
                "max_sequence_length": "INT"},
        outputs={"cond": "COND"},
        description="Wan 文本编码（UMT5-XXL → COND）：输出 prompt_embeds + "
                    "negative_prompt_embeds（Wan 有 CFG，正负两路）。同一 prompt "
                    "多次生成可复用编码；max_sequence_length=0 用 pipe 装配时的值")
    def _op_encode(model, prompt="", negative_prompt="",
                   max_sequence_length=0):
        import torch

        pipe = model
        if not prompt:
            raise ValueError("wan.encode: prompt 必填（Wan 是文本引导模型）")
        if not hasattr(pipe, "encode_prompt"):
            raise ValueError("wan.encode: model 不是 wan.pipe 的输出")
        msl = int(max_sequence_length) or int(
            getattr(pipe, "_wan_max_seq", 512))
        dev = getattr(pipe, "_execution_device", None)
        if dev is None:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        with torch.no_grad():
            pos, neg = pipe.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt or None,
                do_classifier_free_guidance=True,
                num_videos_per_prompt=1,
                max_sequence_length=msl,
                device=dev)
        cond = {"kind": "wan",
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "prompt_embeds": pos.detach().to("cpu"),
                "negative_prompt_embeds": (neg.detach().to("cpu")
                                           if neg is not None else None),
                "num_text_tokens": int(pos.shape[1]),
                "max_sequence_length": msl}
        print(f"[wan] encode tokens={cond['num_text_tokens']} "
              f"neg={'yes' if neg is not None else 'no'} "
              f"{time.time()-t0:.1f}s", flush=True)
        return {"cond": cond}

    @registry.register(
        "wan.gen",
        inputs={"model": "MODEL", "cond": "COND", "prompt": "STRING",
                "negative_prompt": "STRING", "image": "IMAGE",
                "last_image": "IMAGE", "num_frames": "INT",
                "height": "INT", "width": "INT", "steps": "INT",
                "guidance_scale": "FLOAT", "guidance_scale_2": "FLOAT",
                "num_videos_per_prompt": "INT",
                "max_sequence_length": "INT", "fps": "INT",
                "seed": "INT"},
        outputs={"video": "VIDEO", "seed": "INT", "fps": "INT"},
        description="Wan 视频生成：无 image → T2V；给 image → I2V；"
                    "同时给 image+last_image → 首尾帧插值。**Wan 有 CFG**"
                    "（guidance_scale>1 时生效，negative_prompt 才有用），"
                    "与 H3 不同。**建议接 wan-text-encode 的 cond**（复用编码）；"
                    "不接时按 prompt 现场编码。帧数自动对齐 4k+1，fps=16"
                    "（Wan2.1）/24（Wan2.2-TI2V-5B）。无音轨。分钟级任务，请经 /jobs。")
    def _op_gen(model, cond=None, prompt="", negative_prompt="", image=None,
                last_image=None, num_frames=81, height=480, width=832,
                steps=30, guidance_scale=5.0, guidance_scale_2=0.0,
                num_videos_per_prompt=1, max_sequence_length=0, fps=0,
                seed=-1):
        import random

        import numpy as np
        import torch
        from PIL import Image

        pipe = model
        if not hasattr(pipe, "transformer"):
            raise ValueError("wan.gen: model 不是 wan.pipe 的输出")
        pre_cond = None
        if cond is not None:
            if not isinstance(cond, dict) or cond.get("kind") != "wan":
                raise ValueError(
                    "wan.gen: cond 不是 wan 编码产物（须来自 wan.encode）")
            pre_cond = cond
            prompt = prompt or cond.get("prompt") or ""
            if not prompt:
                raise ValueError("wan.gen: cond 缺 prompt 原文")
            if not negative_prompt:
                negative_prompt = cond.get("negative_prompt") or ""
        if not prompt:
            raise ValueError("wan.gen: prompt 必填（Wan 是文本引导模型）")

        def _to_pil(obj):
            """IMAGE 对象 → PIL（兼容 {'frames':[...]} / {'path':...} / PIL）。"""
            if obj is None:
                return None
            if isinstance(obj, Image.Image):
                return obj
            if isinstance(obj, dict):
                frames = obj.get("frames")
                if frames:
                    return frames[0] if isinstance(frames, list) else frames
                p = obj.get("path")
                if p:
                    return Image.open(p).convert("RGB")
            if isinstance(obj, np.ndarray):
                return Image.fromarray(obj.astype("uint8"))
            return obj

        img = _to_pil(image)
        last = _to_pil(last_image)

        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        aligned = _align_frames(num_frames)
        if aligned != num_frames:
            print(f"[wan] num_frames {num_frames} -> {aligned}（4k+1 对齐）",
                  flush=True)
        h, w = int(height), int(width)
        if h <= 0 or w <= 0:
            if img is not None:
                w, h = img.size
            else:
                w, h = 832, 480
        # Wan 空间压缩 8，patch 2 → 边长取 16 的倍数（官方 aspect_ratio_resize）
        mod = 16
        h = max(mod, round(h / mod) * mod)
        w = max(mod, round(w / mod) * mod)

        fps_out = int(fps) if fps and fps > 0 else int(
            getattr(pipe, "_wan_fps", WAN_FPS_DEFAULT))
        msl = int(max_sequence_length) or int(
            getattr(pipe, "_wan_max_seq", 512))
        kwargs = dict(num_frames=aligned, height=h, width=w,
                      num_inference_steps=int(steps),
                      guidance_scale=float(guidance_scale),
                      num_videos_per_prompt=max(1, int(num_videos_per_prompt)),
                      max_sequence_length=msl,
                      generator=gen, output_type="np")
        if pre_cond is not None:
            # 预编码条件：embeds 从对象仓库取回（CPU 存的）→ 上到 pipe 设备；
            # 批量 >1 时按 encode_prompt 的语义沿 batch 维重复。
            _dev = getattr(pipe, "_execution_device", None)
            if _dev is None:
                _dev = torch.device("cuda" if torch.cuda.is_available()
                                    else "cpu")
            try:
                _dt = next(pipe.transformer.parameters()).dtype
            except Exception:  # noqa: BLE001 - 模型无参数时回退 bf16
                _dt = torch.bfloat16
            _n = max(1, int(num_videos_per_prompt))
            pe = pre_cond["prompt_embeds"].to(_dev, _dt)
            ne = pre_cond.get("negative_prompt_embeds")
            ne = ne.to(_dev, _dt) if ne is not None else None
            if _n > 1:
                pe = pe.repeat(1, _n, 1)
                ne = ne.repeat(1, _n, 1) if ne is not None else None
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = None
            kwargs["prompt_embeds"] = pe
            kwargs["negative_prompt_embeds"] = ne
            print(f"[wan] gen 用预编码 cond（tokens="
                  f"{pre_cond.get('num_text_tokens')}）", flush=True)
        else:
            kwargs["prompt"] = prompt
            kwargs["negative_prompt"] = negative_prompt or None
        # MoE 第二阶段 CFG（仅 Wan2.2 双专家生效；0=沿用 guidance_scale）
        if guidance_scale_2 and float(guidance_scale_2) > 0:
            kwargs["guidance_scale_2"] = float(guidance_scale_2)
        t0 = time.time()
        if img is not None:
            # I2V / 首尾帧：WanImageToVideoPipeline 认 image + last_image
            if last is not None:
                kwargs["last_image"] = last
            kwargs["image"] = img
            out = pipe(**kwargs)
        frames_np = out.frames[0]
        # 诊断（真机排障用）：确认 pipeline 输出形状/数值范围是否如预期
        print(f"[wan] frames_np shape={getattr(frames_np, 'shape', '?')} "
              f"dtype={getattr(frames_np, 'dtype', '?')} "
              f"min={float(np.min(frames_np)):.3f} "
              f"max={float(np.max(frames_np)):.3f} "
              f"mean={float(np.mean(frames_np)):.3f}", flush=True)
        # diffusers 的 WanPipeline 在 output_type="np" 下返回
        # (num_frames, height, width, channels)，范围 [0, 1]。若返回 [-1,1]
        # 或通道在前，图像会错乱（真机：黄黑色块/横向条带），故按形状自适应。
        arr = np.asarray(frames_np)
        if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))      # (T,C,H,W) → (T,H,W,C)
        if arr.min() < -0.05:
            arr = (arr + 1.0) / 2.0                     # [-1,1] → [0,1]
        frames = [Image.fromarray(
            (np.clip(f, 0, 1) * 255).round().astype("uint8"))
            for f in arr]
        print(f"[wan] gen done {len(frames)}帧 {fps_out}fps "
              f"{time.time()-t0:.1f}s", flush=True)
        return {"video": {"frames": frames, "fps": fps_out,
                          "num_frames": len(frames)},
                "seed": seed, "fps": fps_out}

    @registry.register(
        "wan.animate",
        inputs={"model": "MODEL", "image": "IMAGE",
                "pose_video": "VIDEO", "face_video": "VIDEO",
                "background_video": "VIDEO", "mask_video": "VIDEO",
                "prompt": "STRING", "negative_prompt": "STRING",
                "mode": "STRING", "segment_frame_length": "INT",
                "prev_segment_conditioning_frames": "INT",
                "motion_encode_batch_size": "INT",
                "height": "INT", "width": "INT", "steps": "INT",
                "guidance_scale": "FLOAT", "seed": "INT"},
        outputs={"video": "VIDEO", "seed": "INT", "fps": "INT"},
        description="Wan-Animate：mode=animate 让角色图按参考视频动作/表情动起来；"
                    "mode=replace 把背景视频里的人换成新角色（需 background_"
                    "video + mask_video：白=生成/黑=保留）。pose_video 与 "
                    "face_video 必须是**预处理后的**骨骼/人脸表示（官方 "
                    "Wan2.2 仓库 preprocess 脚本产出），不能直接喂原始视频。")
    def _op_animate(model, image=None, pose_video=None, face_video=None,
                    background_video=None, mask_video=None,
                    prompt="", negative_prompt="", mode="animate",
                    segment_frame_length=77,
                    prev_segment_conditioning_frames=1,
                    motion_encode_batch_size=0,
                    height=0, width=0, steps=20,
                    guidance_scale=1.0, seed=-1):
        import random

        import numpy as np
        import torch
        from PIL import Image as _Image

        pipe = model
        mode = (mode or "animate").lower()
        if mode not in ("animate", "replace"):
            raise ValueError("wan.animate: mode 须为 animate / replace")
        if image is None:
            raise ValueError("wan.animate: image（角色参考图）必填")
        if mode == "replace" and (background_video is None
                                  or mask_video is None):
            raise ValueError("wan.animate: replace 模式需 background_video "
                             "与 mask_video（白=生成区/黑=保留区）")

        def _pils(obj):
            if obj is None:
                return None
            if isinstance(obj, _Image.Image):
                return [obj]
            if isinstance(obj, dict):
                if obj.get("frames"):
                    return list(obj["frames"])
                if obj.get("path"):
                    return obj["path"]  # diffusers load_video 支持路径
            return obj

        def _frame_array(obj):
            """VIDEO → (N,H,W,3) float[0,1] ndarray（diffusers 期望形态）。"""
            pils = _pils(obj)
            if isinstance(pils, str):
                from diffusers.utils import load_video
                arr = load_video(pils)
                return arr
            if isinstance(pils, list) and pils and isinstance(
                    pils[0], _Image.Image):
                # 统一成 float32 [0,1]：diffusers 各 pipe 对 uint8/float 的
                # 容忍度不一致，显式归一化最稳（PIL 输入内部也是这么转的）
                return [np.asarray(f.convert("RGB"), dtype=np.float32) / 255.0
                        for f in pils]
            return pils

        ref = _pils(image)
        ref = ref[0] if isinstance(ref, list) else ref
        pose = _frame_array(pose_video)
        face = _frame_array(face_video)
        bg = _frame_array(background_video) if background_video is not None else None
        msk = _frame_array(mask_video) if mask_video is not None else None

        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        h, w = int(height), int(width)
        if h <= 0 or w <= 0:
            w, h = ref.size
        mod = 16
        h = max(mod, round(h / mod) * mod)
        w = max(mod, round(w / mod) * mod)

        fps = int(getattr(pipe, "_wan_fps", 30))
        kwargs = dict(image=ref, pose_video=pose, face_video=face,
                      prompt=prompt, negative_prompt=negative_prompt or None,
                      height=h, width=w,
                      segment_frame_length=int(segment_frame_length),
                      guidance_scale=float(guidance_scale),
                      num_inference_steps=int(steps),
                      mode=mode, generator=gen, output_type="np")
        if int(prev_segment_conditioning_frames) >= 0:
            # 段间时序引导（diffusers 默认 1；5 更连贯但更吃显存）
            kwargs["prev_segment_conditioning_frames"] = int(
                prev_segment_conditioning_frames)
        if int(motion_encode_batch_size) > 0:
            # 动作 VAE encode 的 batch（大分辨率/长片段时调低可省显存）
            kwargs["motion_encode_batch_size"] = int(motion_encode_batch_size)
        if mode == "replace":
            kwargs["background_video"] = bg
            kwargs["mask_video"] = msk
        t0 = time.time()
        out = pipe(**kwargs)
        frames_np = out.frames[0]
        # 与 wan.gen 同款自适应：diffusers 的 output_type="np" 一般给
        # (T,H,W,C)/[0,1]，但版本间可能给 (T,C,H,W) 或 [-1,1]，
        # 不处理会得到横向条带/全黑（真机在 wan.gen 上踩过）。
        arr = np.asarray(frames_np)
        print(f"[wan] animate frames shape={arr.shape} "
              f"min={float(arr.min()):.3f} max={float(arr.max()):.3f}",
              flush=True)
        if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.min() < -0.05:
            arr = (arr + 1.0) / 2.0
        frames = [_Image.fromarray(
            (np.clip(f, 0, 1) * 255).round().astype("uint8")) for f in arr]
        print(f"[wan] animate({mode}) done {len(frames)}帧 {fps}fps "
              f"{time.time()-t0:.1f}s", flush=True)
        return {"video": {"frames": frames, "fps": fps,
                          "num_frames": len(frames)},
                "seed": seed, "fps": fps}



    @registry.register(
        "wan.lora_load",
        inputs={"model": "MODEL", "lora": "STRING", "lora_scale": "FLOAT",
                "relight_lora": "STRING", "relight_scale": "FLOAT"},
        outputs={"lora_config": "STRING"},
        description="Wan LoRA 统一入口（配置校验 + 归一化）。lora 支持"
                    "逗号分隔多挂，每项可带 @强度：'a.safetensors:1.0,"
                    "b.safetensors:0.8'。relight_lora 是官方打光 LoRA"
                    "（让替换人物的光照匹配背景，治'像另一个图层'），"
                    "**替换任务建议必开**。输出配置串给 wan-animate2 的 "
                    "lora / relight_lora 入口；真正合并发生在装配后"
                    "（权重还在 CPU 内存态时，与块级 offload 无冲突）。")
    def _op_lora_load(model=None, lora="", lora_scale=1.0,
                      relight_lora="", relight_scale=1.0):
        def _check(spec, scale, label):
            items = []
            for raw in str(spec or "").split(","):
                raw = raw.strip()
                if not raw:
                    continue
                path, _, st = raw.rpartition(":")
                if not path or "/" not in path:
                    path, st = raw, ""
                try:
                    sc = float(st) if st else float(scale)
                except ValueError:
                    path, sc = raw, float(scale)
                p = path if os.path.isabs(path) else os.path.join(
                    os.environ.get("MODELS_DIR", "/models"), "loras", path)
                if not os.path.isfile(p):
                    raise FileNotFoundError(
                        f"wan.lora_load: {label} 不存在: {p}")
                items.append(f"{p}:{sc:g}")
            return items
        lor = _check(lora, lora_scale, "lora")
        rel = _check(relight_lora, relight_scale, "relight_lora")
        cfg = "|".join(["LORA=" + ",".join(lor),
                        "RELIGHT=" + ",".join(rel)])
        print(f"[wan.lora_load] lora={len(lor)} 项 relight={len(rel)} 项 "
              f"-> {cfg[:160]}", flush=True)
        return {"lora_config": cfg}

    @registry.register(
        "wan.te_cache",
        inputs={"prompt": "STRING", "negative_prompt": "STRING",
                "max_sequence_length": "INT", "text_encoder": "STRING"},
        outputs={"embeds_path": "STRING", "negative_embeds_path": "STRING",
                 "cache_hit": "BOOL"},
        description="UMT5-XXL 文本向量缓存（省 11GB 模型加载）。首次：加载 "
                    "文本编码器 → 编码 prompt/negative → 存 safetensors；"
                    "命中缓存：直接返回路径，**完全不加载文本模型**。输出给 "
                    "wan-animate2 的 prompt_embeds_path / negative_embeds_path。"
                    "同一 prompt 反复换人/换视频时收益最大。")
    def _op_te_cache(prompt="", negative_prompt="", max_sequence_length=0,
                     text_encoder=""):
        import hashlib
        import torch as _t
        from safetensors.torch import load_file, save_file
        msl = int(max_sequence_length or 0) or 512
        te_dir = text_encoder or os.environ.get(
            "WAN_ANIMATE_TE",
            f"{os.environ.get('MODELS_DIR', '/models')}/wan/config/text_encoder")
        tok_dir = os.path.join(os.path.dirname(te_dir.rstrip("/")), "tokenizer")
        # 版本号参与 key：编码逻辑一变（如截断+补零修复）旧缓存自动失效，
        # 否则会静默复用错误向量继续出废片。
        _TE_CACHE_VER = "v2-truncpad"
        key = hashlib.sha256(
            ("\x1f".join([_TE_CACHE_VER, str(prompt), str(negative_prompt),
                           str(msl), os.path.basename(te_dir)])).encode("utf-8")
        ).hexdigest()[:24]
        cdir = os.path.join(os.environ.get("MODELS_DIR", "/models"),
                            ".cache", "wan_te")
        os.makedirs(cdir, exist_ok=True)
        fpos = os.path.join(cdir, f"{key}_pos.safetensors")
        fneg = os.path.join(cdir, f"{key}_neg.safetensors")
        if os.path.isfile(fpos):
            print(f"[wan.te_cache] 命中缓存 {key}", flush=True)
            return {"embeds_path": fpos, "negative_embeds_path": fneg,
                    "cache_hit": True}

        from transformers import AutoTokenizer, UMT5EncoderModel
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(tok_dir)
        te = UMT5EncoderModel.from_pretrained(
            te_dir, torch_dtype=_t.bfloat16)
        dev = "cuda" if _t.cuda.is_available() else "cpu"
        te = te.to(dev).eval()

        def _enc(text):
            # **必须复刻 pipeline 的截断+补零（2026-11-16 雪花复盘）**：
            # WanAnimatePipeline._get_t5_prompt_embeds 取 last_hidden_state
            # 后按 mask 求实际 token 数、截断、再用 **0** 补到 max_sequence_length。
            # 直接存 last_hidden_state 会把 padding 区留成"模型对 pad token 的
            # 编码"（非 0）→ 条件与训练分布不符 → 生成物化成一团彩色碎片。
            ids = tok(text or "", return_tensors="pt", padding="max_length",
                      max_length=msl, truncation=True).to(dev)
            with _t.no_grad():
                e = te(**ids).last_hidden_state
            seq_lens = ids["attention_mask"].gt(0).sum(dim=1).long()
            rows = [e[i, :int(v)] for i, v in enumerate(seq_lens)]
            e = _t.stack(
                [_t.cat([r, r.new_zeros(msl - r.size(0), r.size(1))])
                 for r in rows], dim=0)
            return e.to("cpu", dtype=_t.bfloat16)

        pos = _enc(prompt)
        neg = _enc(negative_prompt or "")
        save_file({"embeds": pos}, fpos)
        save_file({"embeds": neg}, fneg)
        del te, tok
        _t.cuda.empty_cache()
        print(f"[wan.te_cache] 编码完成并落盘 {key} "
              f"shape={tuple(pos.shape)} {time.time()-t0:.1f}s "
              f"（文本模型已卸载）", flush=True)
        return {"embeds_path": fpos, "negative_embeds_path": fneg,
                "cache_hit": False}

    @registry.register(
        "wan.animate2",
        inputs={"model": "MODEL", "image": "IMAGE", "pose_video": "VIDEO",
                "face_video": "VIDEO", "background_video": "VIDEO",
                "mask_video": "VIDEO", "mode": "STRING",
                "prompt": "STRING", "negative_prompt": "STRING",
                "pose_strength": "FLOAT", "pose_start_percent": "FLOAT",
                "pose_end_percent": "FLOAT",
                "reference_image_strength": "FLOAT",
                "segment_frame_length": "INT", "height": "INT",
                "width": "INT", "steps": "INT", "guidance_scale": "FLOAT",
                "seed": "INT", "fps": "INT",
                "inpaint_fill": "STRING", "inpaint_erode": "INT",
                "inpaint_expand": "INT", "inpaint_telea_radius": "INT",
                "inpaint_ring_width": "INT",
                "composite": "BOOL", "composite_expand": "INT",
                "composite_feather": "INT", "composite_gain": "FLOAT",
                "composite_dehaze": "BOOL",
                "composite_fill_black": "FLOAT",
                "composite_edge_clean": "INT",
                "composite_unpremul": "FLOAT",
                "mask_invert": "INT",
                "latent_inpaint": "INT", "inpaint_dim": "FLOAT",
        "face_from_ref": "INT",
                "composite_dehaze_strength": "FLOAT",
                "lora": "STRING", "relight_lora": "STRING",
                "relight_scale": "FLOAT",
                "prompt_embeds_path": "STRING",
                "negative_embeds_path": "STRING"},
        outputs={"video": "VIDEO", "seed": "INT", "fps": "INT"},
        description="Wan-Animate 角色动画迁移（**WanAnimatePipeline 路线**，"
                    "2026-11-14 重构）：角色图 image（自动 pad 保比例）+ "
                    "DWPose 骨架视频 pose_video（video-control-preprocess "
                    "mode=pose 产出）→ 角色按骨架运动。face_video 可选"
                    "（面部表情 crop，缺省全黑=面部身份完全交给参考图）。"
                    "显存：apply_group_offloading 块级，24GB 实测峰值 17.5GB。"
                    "pose_strength 等 ComfyUI 旋钮在此实现中不适用（保留参数"
                    "兼容旧工作流，不参与计算）。")
    def _op_animate2(model=None, image=None, pose_video=None, face_video=None,
                     background_video=None, mask_video=None, mode="auto",
                     inpaint_fill="black", inpaint_erode=0,
                     inpaint_expand=0, inpaint_telea_radius=5,
                     inpaint_ring_width=10,
                     prompt="", negative_prompt="", pose_strength=1.0,
                     pose_start_percent=0.0, pose_end_percent=1.0,
                     reference_image_strength=1.0, segment_frame_length=81,
                     height=0, width=0, steps=20, guidance_scale=1.0,
                     seed=-1, fps=0,
                     composite=False, composite_expand=8, composite_feather=32,
                     composite_gain=1.0, composite_dehaze=True,
                     composite_dehaze_strength=0.95,
                     composite_fill_black=0.0, composite_edge_clean=0,
                     composite_unpremul=0.0, mask_invert=0,
                     latent_inpaint=0, inpaint_dim=0.38,
                     face_from_ref=1,
                     lora="", relight_lora="",
                     relight_scale=1.0, prompt_embeds_path="",
                     negative_embeds_path=""):
        import random
        mk_out = None            # replace 分支填充（composite 合成的软 mask 源）
        _bg_orig_frames = None   # replace 分支填充（未抹除的原始背景）

        import numpy as np
        import torch
        from PIL import Image as _Image

        # **独立装配（WAN_STANDALONE=1，默认开）**：照抄官方脚本装配，
        # 不经 _op_pipe 的组件缓存。WAN_STANDALONE=0 回到旧行为（用 model）。
        # LoRA 配置：优先用上游 wan-lora-load 的 lora_config，否则用本节点参数
        _lora_cfg = str(lora or "").strip()
        if _lora_cfg and "LORA=" not in _lora_cfg:
            _rel = str(relight_lora or "").strip()
            _lora_cfg = "LORA=" + _lora_cfg + "|RELIGHT=" + (
                f"{_rel}:{float(relight_scale):g}" if _rel else "")
        elif not _lora_cfg and str(relight_lora or "").strip():
            _lora_cfg = ("LORA=|RELIGHT="
                         f"{str(relight_lora).strip()}:{float(relight_scale):g}")
        if _lora_cfg:
            print(f"[wan] lora 配置: {_lora_cfg[:150]}", flush=True)
        if os.environ.get("WAN_STANDALONE", "1") == "1":
            _mroot = os.environ.get("MODELS_DIR", "/models")
            pipe = _assemble_animate_standalone(
                transformer_dir=os.environ.get(
                    "WAN_ANIMATE_DIR",
                    f"{_mroot}/wan-animate-diffusers/transformer"),
                te_dir=os.environ.get(
                    "WAN_ANIMATE_TE", f"{_mroot}/wan/config/text_encoder"),
                vae_file=os.environ.get(
                    "WAN_ANIMATE_VAE",
                    f"{_mroot}/wan/vae/wan_2.1_vae.safetensors"),
                img_enc_file=os.environ.get(
                    "WAN_ANIMATE_CLIP",
                    f"{_mroot}/wan/image_encoders/clip_vision_h.safetensors"),
                flow_shift=float(os.environ.get("WAN_FLOW_SHIFT", "5.0")),
                te_on_cuda=os.environ.get("WAN_TE_ON_CUDA", "0") == "1",
                skip_te=bool(prompt_embeds_path and
                             os.path.isfile(prompt_embeds_path)),
                prompt_embeds_path=prompt_embeds_path,
                negative_embeds_path=negative_embeds_path,
                lora_config=_lora_cfg,
            )
        else:
            pipe = model
        if image is None:
            raise ValueError("wan.animate2: image（角色参考图）必填")
        if pose_video is None:
            raise ValueError(
                "wan.animate2: pose_video 必填——建议用 video-control-preprocess"
                "（mode=pose）把原视频转成骨架视频再喂进来")

        def _pils(obj):
            if obj is None:
                return None
            if isinstance(obj, _Image.Image):
                return [obj]
            if isinstance(obj, dict):
                if obj.get("frames"):
                    return list(obj["frames"])
                if obj.get("path"):
                    from diffusers.utils import load_video
                    return load_video(obj["path"])
            return obj

        ref = _pils(image)
        ref = ref[0] if isinstance(ref, list) else ref
        pose = _pils(pose_video)

        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        h, w = int(height), int(width)
        if h <= 0 or w <= 0:
            # **缺省尺寸必须取 pose_video（动作源）而非参考图**
            # （2026-11-14 真机：取了竖版参考图 450×700，横版 896×512 骨架
            # 被纵向挤压 → 人物扭曲独舞、画幅全错）。参考图随后 pad 进
            # 这个画幅即可，动作与画幅永远跟随源视频。
            pw, ph = pose[0].size
            w, h = pw, ph
        mod = 16
        h = max(mod, round(h / mod) * mod)
        w = max(mod, round(w / mod) * mod)

        # **fps 跟随源视频（2026-11-15 用户验收标准：时长/动作速度与源
        # 一致）**：写死 Wan 训练 16fps 会把 24fps 源片慢放 1.5 倍
        # （61帧@16=3.8s vs 源 2.54s）。fps=0 自动：pose_video 对象 fps →
        # path 元数据 → Wan 默认 16。
        fps_req = int(fps)
        if fps_req <= 0 and isinstance(pose_video, dict):
            fps_req = int(float(pose_video.get("fps") or 0))
            if fps_req <= 0 and pose_video.get("path"):
                try:
                    import cv2 as _cv2
                    _cap = _cv2.VideoCapture(pose_video["path"])
                    fps_req = int(round(_cap.get(_cv2.CAP_PROP_FPS) or 0))
                    _cap.release()
                except Exception:  # noqa: BLE001
                    fps_req = 0
        fps = fps_req if fps_req > 0 else int(
            getattr(pipe, "_wan_fps", WAN_FPS_DEFAULT))
        # **2026-11-14 路线纠错（WanAnimatePipeline，真机出片验证配方）**：
        # 弃用 WanAnimate2ModularPipeline（driving_video 参数 / 假 hook 干扰 /
        # 22GB 不明占用三连坑）。新调用约定：
        #   image       角色参考图（**pad 保持比例**，直接 resize 会把竖版人
        #               拉成扁人，VAE 条件失效 → 外观不跟随，exec 1405~1411 教训）
        #   pose_video  DWPose 骨架渲染帧（pipeline 内部 VAE 编码成 pose latent）
        #   face_video  面部 crop；239 工作流暂无 face 节点 → 全黑帧
        #               （motion_encoder 输出零向量，面部身份完全交给参考图，
        #               2026-11-14 实验 E 验证可行）
        #   mode        "animate"（动画迁移；replace 人物替换需另传 bg/mask，
        #               且 bg 必须 ×(1-mask) 抹掉原人——同日出片验证的根因修复）
        # offload 已在 _op_pipe 装配时挂好（apply_group_offloading 块级），
        # 这里不重复挂（重复挂 hook 会叠加 pre_forward，搬运两次直接 OOM）。
        mod = 16
        h = max(mod, round(h / mod) * mod)
        w = max(mod, round(w / mod) * mod)

        # 参考图 pad 到目标比例（灰边 114 是 ImageNet 均值，比黑边温和）
        if isinstance(ref, _Image.Image) and ref.size != (w, h):
            iw, ih = ref.size
            scale = min(w / iw, h / ih)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            canvas = _Image.new("RGB", (w, h), (114, 114, 114))
            canvas.paste(ref.resize((nw, nh), _Image.BILINEAR),
                         ((w - nw) // 2, (h - nh) // 2))
            ref = canvas
            print(f"[wan] animate2 参考图 pad {iw}x{ih} -> {w}x{h}", flush=True)

        # face 全黑（239 无 face 节点；有 face_video 输入时优先生效）
        face = _pils(face_video) if face_video is not None else None
        # **face 改为参考图人脸（2026-11-18 v36，治原人身份泄漏/鬼影）**：
        # Wan-Animate 的 face_video 是**面部身份驱动通道**——喂原视频人脸
        # 则生成的脸=原人的脸（v35 实证：脸/发型与原视频完全一致，即
        # "双重曝光鬼影"的主因之一）。替换任务必须让身份来自参考图：
        # face_from_ref>0 时（默认开）从 image（参考图）裁上部人脸区，
        # 重复到全部帧 → 身份=参考图、表情不跟随（舞蹈场景可接受）。
        if int(face_from_ref or 0) > 0:
            _rw, _rh = ref.size
            _fy0 = int(0.02 * _rh)
            _fy1 = int(0.22 * _rh)
            _fs = max(32, _fy1 - _fy0)
            _fx0 = max(0, (_rw - _fs) // 2)
            _fcrop = ref.crop((_fx0, _fy0, min(_rw, _fx0 + _fs), _fy1)
                              ).resize((512, 512), _Image.LANCZOS)
            face = [_fcrop for _ in pose]
            print(f"[wan] animate2 face 来自参考图（{_fs}px 裁剪→512²，"
                  f"{len(face)} 帧）——身份由参考图驱动", flush=True)
        if face is None:
            face = [_Image.new("RGB", (512, 512), (0, 0, 0))
                    for _ in pose]
            print(f"[wan] animate2 face_video 缺省 → 全黑 {len(face)} 帧",
                  flush=True)

        # **逐步 NaN 监测（2026-11-14）**：同配置间歇全 NaN（1413 健康 /
        # 1416/1417 炸），需要定位 NaN 首现步号与段号（双段续接嫌疑）。
        try:
            from app import execution as _exec
            _job = _exec.current()
        except Exception:  # noqa: BLE001
            _job = None
        if _job is not None:
            _job.set_progress(0, int(steps))
        _last_pct = [-1]

        # **latent inpainting 上下文（2026-11-17，diffusers 官方 callback 扩展点）**
        # 只重绘 mask 内，mask 外每步拉回「原视频 latent + 当前 σ 的噪声」——
        # 这样背景像素级保留（不靠涂黑、不靠输出端合成），人物区自由生成。
        # 这正是 H3 inpaint 用的技术，但 Wan 侧走官方的
        # `callback_on_step_end`（pipeline 会把返回的 latents 替换进循环），
        # **完全不绕开 diffusers**（不自己写采样循环）。
        _li_ctx = {}

        def _nan_cb(pipe_, step_idx, timestep, cb_kwargs):
            lat = cb_kwargs.get("latents")
            if lat is not None and bool(torch.isnan(lat).any()):
                print(f"[wan][nan] ⚠ step {step_idx} latents 出现 NaN "
                      f"(t={float(timestep):.1f})", flush=True)
            # ---- latent inpainting ----
            if step_idx == 0:
                print(f"[wan][latent-inpaint] cb 第0步 lat="
                      f"{None if lat is None else tuple(lat.shape)} "
                      f"ctx={'有' if _li_ctx else '空'}", flush=True)
            if _li_ctx and lat is not None:
                try:
                    # σ 用 scheduler 真实值（UniPC/FlowMatch 有 shift，t/1000
                    # 在低步段偏差大 → 拉回的噪声水平不匹配会产生不一致）
                    try:
                        _sig = float(pipe_.scheduler.sigmas[int(step_idx)])
                    except Exception:  # noqa: BLE001
                        _sig = float(timestep) / 1000.0
                    _src = _li_ctx["src"]        # 原视频 latent（未加噪）
                    _nz = _li_ctx["noise"]       # 固定初始噪声
                    _mk = _li_ctx["mask"]        # 人物=1 / 背景=0
                    # **时间维动态对齐（2026-11-17 实测）**：armed 时按
                    # (T_px-1)/4+1 估的 latent 帧数（21帧→6），但 pipeline 内部
                    # 会把视频 pad 到 25 帧（latent=7）——src/noise/mask 按
                    # pipeline 实际 T 最近邻重采样（误差≤1 帧，背景拉回够用）
                    _t = int(lat.shape[2])
                    if int(_src.shape[2]) != _t:
                        _idx = torch.linspace(
                            0, int(_src.shape[2]) - 1, _t).long().to(_src.device)
                        _src = _src[:, :, _idx]
                        _nz = _nz[:, :, _idx]
                        _idxm = torch.linspace(
                            0, int(_mk.shape[2]) - 1, _t).long().to(_mk.device)
                        _mk = _mk[:, :, _idxm]
                    _mix = _src * (1.0 - _sig) + _nz * _sig
                    lat = lat * _mk + _mix * (1.0 - _mk)
                    cb_kwargs["latents"] = lat
                    if step_idx == 0 or step_idx % 4 == 0:
                        print(f"[wan][latent-inpaint] step {step_idx} "
                              f"σ={_sig:.3f} 背景已拉回", flush=True)
                except Exception as _el:  # noqa: BLE001
                    print(f"[wan][latent-inpaint] 跳过（{type(_el).__name__}: "
                          f"{_el}）", flush=True)
            # **画布步数进度**（2026-11-15 用户反馈节点不显示步数）：
            # 经 job.set_progress 推到 wait_job 的 progress 事件流
            if _job is not None:
                try:
                    _job.set_progress(int(step_idx) + 1, int(steps))
                except Exception:  # noqa: BLE001
                    pass
            return cb_kwargs
        # **replace 模式（人物替换，2026-11-14 真机验证配方）**：
        # 给了 background_video + mask_video 即启用——保留原视频背景，
        # 只重绘遮罩区域的人物（成片 238：金发绿裙替换白衣仙女，
        # 黑衣人/山水/能量特效全保留）。
        # ⚠ 关键预处理（根因修复）：diffusers 把**含原人物的完整 bg**
        # 直接 VAE 编码进条件（mask 只走标记通道），原人会压过参考图；
        # 官方训练口径是 **bg × (1-mask) 像素级抹除被替换者**，必须自己做。
        _mode = str(mode or "auto").lower()
        bg = _pils(background_video)
        mk = _pils(mask_video)
        if _mode == "auto":
            _mode = "replace" if (bg and mk) else "animate"
        extra = {}
        if _mode == "replace":
            if not bg or not mk:
                raise ValueError(
                    "wan.animate2: replace 模式必须同时给 background_video "
                    "（原视频）与 mask_video（白=待替换人物）")
            from PIL import ImageFilter as _IF
            n = len(pose)
            bg = (bg * ((n // len(bg)) + 1))[:n]
            mk = (mk * ((n // len(mk)) + 1))[:n]
            bg_masked, mk_out = [], []
            _bg_orig_frames = []
            for b, m in zip(bg, mk):
                b = b.convert("RGB").resize((w, h), _Image.BILINEAR)
                m = m.convert("L").resize((w, h), _Image.BILINEAR)
                # **抹除填充与黑边治理（2026-11-15 对照实验 242 修正）**：
                # blur 填充（241→242 唯一变更）导致模型把 mask 区当"已有
                # 内容"做半透明叠加（人物幽灵化）——回退 black 实心抹除；
                # 黑边改由 inpaint_erode 收缩**合成 mask** 裁掉边缘残留
                # （erode 在羽化前做，收缩后再羽化保持接缝自然）。
                mnp0 = np.asarray(m, dtype=np.float32) / 255.0
                if int(inpaint_erode) > 0:
                    import cv2 as _cv2
                    _k = int(inpaint_erode)
                    _kernel = _cv2.getStructuringElement(
                        _cv2.MORPH_ELLIPSE, (_k * 2 + 1, _k * 2 + 1))
                    mnp0 = _cv2.erode(mnp0, _kernel)
                m = _Image.fromarray((mnp0 * 255).astype("uint8"))
                # 羽化 16px（v34 从 8 加宽）：telea_dim 的亮度台阶需要更宽的
                # 平缓过渡，8px 时 VAE/DiT 仍能感知到暗区边界（残 Halo）
                m = m.filter(_IF.GaussianBlur(radius=16))
                mnp = np.asarray(m, dtype=np.float32) / 255.0
                # **抹黑区与合成区分离（2026-11-16「像贴图层+黑边」根治）**：
                # 原实现两个用途共用一个 mask → 模型只能在**紧贴人物轮廓的
                # 黑色边界**内生成，人物边缘必然被染暗；composite 的宽过渡带
                # 又把这份暗色混回画面 → 一圈黑描边、人物像另一图层。
                # 现在：抹黑用**外扩 inpaint_expand** 的更大区域（模型有充足
                # 空间画出完整轮廓，边缘不碰黑），composite 仍用精确 mask。
                _iexp = int(inpaint_expand or 0)
                if _iexp > 0:
                    import cv2 as _cv2b
                    _kb = _cv2b.getStructuringElement(
                        _cv2b.MORPH_ELLIPSE, (_iexp * 2 + 1, _iexp * 2 + 1))
                    _mnp_wide = _cv2b.dilate(mnp, _kb)
                else:
                    _mnp_wide = mnp
                orig = np.asarray(b, dtype=np.float32)
                _fill = str(inpaint_fill)
                if _fill == "blur":
                    blur = np.asarray(b.filter(_IF.GaussianBlur(radius=24)),
                                      dtype=np.float32)
                    wiped = np.clip(orig * (1.0 - _mnp_wide[..., None])
                                    + blur * _mnp_wide[..., None], 0, 255)
                elif _fill in ("gray", "bgmean"):
                    # 灰洞/背景均值色：黑洞语义 → 中性色，减轻生成期边缘染暗
                    _fc = (orig[(_mnp_wide <= 0.5)].mean(axis=0)
                           if _fill == "bgmean" and (_mnp_wide <= 0.5).any()
                           else np.array([128.0, 128.0, 128.0]))
                    wiped = np.clip(orig * (1.0 - _mnp_wide[..., None])
                                    + _fc[None, None, :] * _mnp_wide[..., None],
                                    0, 255)
                elif _fill == "telea_ring":
                    # **边缘环带纹理引导（2026-11-16 VAE bleed 最终方案）**：
                    # 纯黑抹除 → 模型参考周围黑色，在人物身上画出深色衣物
                    # （bleed 表现为"内容染色"）；Telea 全量填充 → 模型看到
                    # 一片幕布纹理、失去"这里要画人"的结构锚点 → 生成崩坏。
                    # 折中：**只把 mask 边缘一圈（ring_width px）换成背景纹理**，
                    # 中心保持黑洞。模型边缘接壤的是幕布/桌面纹理（不再被黑
                    # 染色），中心仍有明确"待生成"信号。
                    import cv2 as _cv2r
                    _m = (_mnp_wide > 0.5).astype(np.uint8)
                    _rw = max(1, int(inpaint_ring_width or 10))
                    _ker = _cv2r.getStructuringElement(
                        _cv2r.MORPH_ELLIPSE, (_rw * 2 + 1, _rw * 2 + 1))
                    _inner = _cv2r.erode(_m, _ker)
                    _ring = _m - _inner
                    _black = np.clip(orig * (1.0 - _mnp_wide[..., None]), 0, 255)
                    _tex = _cv2r.inpaint(
                        orig.astype(np.uint8), _m * 255,
                        max(1, int(inpaint_telea_radius or 5)),
                        _cv2r.INPAINT_TELEA).astype(np.float32)
                    wiped = np.where(_ring[..., None] > 0, _tex, _black)
                elif _fill == "telea":
                    # **Telea 纹理延展（2026-11-16 VAE bleed 治法）**：
                    # 黑色抹除会在潜空间边缘产生 bleed（黑边污染的物理源头），
                    # blur/gray/bgmean 又会被模型当成"已存在内容"重绘成雾态
                    # （历史实测证伪）。Telea 是**沿 mask 边缘法线把背景纹理
                    # 向内推进**，模型边缘接壤的是符合透视的幕布/桌面纹理，
                    # 即便有轻微溢出，溢出的也是背景自己的颜色。
                    import cv2 as _cv2t
                    _m255 = ((_mnp_wide > 0.5).astype(np.uint8)) * 255
                    _r = int(inpaint_telea_radius or 5)
                    wiped = _cv2t.inpaint(orig.astype(np.uint8), _m255,
                                          max(1, _r), _cv2t.INPAINT_TELEA
                                          ).astype(np.float32)
                elif _fill == "telea_dim":
                    # **暗化全填（2026-11-18 v33，幽灵↔Halo 两难的中间态）**：
                    # 全填亮窗帘（v31）→ 条件覆盖，人物幽灵化；黑洞核心（v32）
                    # → 人物实心但环带内缘断崖显影成 Halo。中间态：telea 全填
                    # 保证纹理连续（VAE 无断崖），mask 区整体压暗到 45%——暗区
                    # 仍向模型传递"此区域待重绘"的信号，但不制造任何边缘。
                    import cv2 as _cv2d
                    _m255d = ((_mnp_wide > 0.5).astype(np.uint8)) * 255
                    _texd = _cv2d.inpaint(
                        orig.astype(np.uint8), _m255d, 15,
                        _cv2d.INPAINT_TELEA).astype(np.float32)
                    _dim = _texd * (1.0 - min(0.9, max(0.05,
                        float(inpaint_dim if inpaint_dim is not None
                              else 0.38))))
                    wiped = np.clip(
                        orig * (1.0 - _mnp_wide[..., None])
                        + _dim * _mnp_wide[..., None], 0, 255)
                elif _fill == "orig":
                    # **原样背景（2026-11-17 决定性发现）**：diffusers 的
                    # `prepare_prev_segment_cond_latents` 在 task=replace 时直接
                    # `prev_segment_cond_video = background_video[...]`（原样，
                    # 不涂黑），ComfyUI 也是 `image[...] = background_video`
                    # 原样。二者都把**完整背景**作为条件 latent 交给模型，
                    # 配合 4 通道 concat mask 告诉模型「哪里保留」——这才是
                    # 标准 inpainting 语义。我们此前把背景**涂黑**再传，模型
                    # 根本看不到背景，只能在边缘"延续黑洞的暗调" → 黑边。
                    # 本模式直接传原帧，由 mask 通道负责"保留/重绘"的分工。
                    wiped = orig.copy()
                else:  # black（默认，实心替换；用外扩后的宽 mask）
                    wiped = np.clip(orig * (1.0 - _mnp_wide[..., None]), 0, 255)
                bg_masked.append(_Image.fromarray(wiped.astype("uint8")))
                mk_out.append(m)
                _bg_orig_frames.append(np.asarray(b, dtype=np.float32))
            # **mask 极性（2026-11-17 实测）**：diffusers 在 task=replace 里
            # 先做 `mask_video = 1 - mask_video`，再 `get_i2v_mask` 里把
            # 「条件帧」置 1（1=保留原 latent / 0=重绘）。实测传「人物=1」时
            # 成片把原人物原样保留（说明该路径下 1 被当成"保留"），需要反转。
            # mask_invert=1 → 传「人物=0 / 背景=1」。
            if int(mask_invert or 0) > 0:
                from PIL import ImageOps as _IO
                mk_out = [_IO.invert(_m.convert("L")) for _m in mk_out]
                print("[wan] mask 已反相（人物=0 / 背景=1）", flush=True)
            # **诊断打印（2026-11-17）**：确认实际喂给 pipeline 的 bg/mask 是什么
            # （黑边根因排查收敛到"bg 该原样还是涂黑 + mask 极性"两点，必须看得见）
            try:
                _m0 = np.asarray(mk_out[0], dtype=np.float32) / 255.0
                _b0 = np.asarray(bg_masked[0], dtype=np.float32)
                _orig0 = np.asarray(bg_frames[0] if 'bg_frames' in dir() else bg_masked[0],
                                    dtype=np.float32)
                print(f"[wan] 传给 pipeline: mask(第0帧) mean={_m0.mean():.3f} "
                      f"人物像素={float((_m0 > 0.5).mean()) * 100:.1f}% | "
                      f"bg(第0帧) mean={_b0.mean():.1f} min={_b0.min():.0f} "
                      f"max={_b0.max():.0f}", flush=True)
            except Exception:  # noqa: BLE001
                pass
            extra = dict(background_video=bg_masked, mask_video=mk_out)
            print(f"[wan] animate2 replace: fill={inpaint_fill} "
                  f"erode={inpaint_erode}px 羽化8px ({len(bg_masked)}帧)",
                  flush=True)

        # **预计算文本向量（2026-11-16）**：有缓存就直传 prompt_embeds，
        # pipeline 的 `if prompt_embeds is None` 不成立 → 完全不走 UMT5。
        if prompt_embeds_path and os.path.isfile(prompt_embeds_path):
            from safetensors.torch import load_file as _lf
            _pb = _lf(prompt_embeds_path)["embeds"].to("cuda")
            _nb = None
            if negative_embeds_path and os.path.isfile(negative_embeds_path):
                _nb = _lf(negative_embeds_path)["embeds"].to("cuda")
            print(f"[wan] 使用预计算文本向量 shape={tuple(_pb.shape)}"
                  f"（跳过 UMT5 编码）", flush=True)
        else:
            _pb = _nb = None

        # **准备 latent inpainting 上下文（原视频 latent + mask latent + 固定噪声）**
        print(f"[wan][li-diag] latent_inpaint={latent_inpaint!r} "
              f"bg_orig={'None' if _bg_orig_frames is None else len(_bg_orig_frames)} "
              f"mk={'None' if mk_out is None else len(mk_out)}", flush=True)
        if int(latent_inpaint or 0) > 0 and _bg_orig_frames is not None \
                and mk_out is not None:
            try:
                import cv2 as _cv2li
                import torch as _tj
                _vae = getattr(pipe, "vae", None)
                if _vae is not None:
                    # **拉回源=填充版背景（2026-11-18 v29，治半透明幽灵）**：
                    # 原用 _bg_orig_frames（原视频，含原人）→ 软 alpha 边缘/低置信
                    # 区的拉回权重把**原人 latent** 混进生成 → 人物半透明（v28
                    # 实证）。改用 bg_masked（inpaint_fill 填充后的干净背景）：
                    # ① 人物边缘拉回的是窗帘/填充纹理，无原人幽灵；
                    # ② 条件与拉回源一致（都在 bg_masked 上），生成与拉回不打架；
                    # ③ mask 外区域 bg_masked==原视频，背景照旧像素级保真。
                    _src_frames = (bg_masked if isinstance(bg_masked, list)
                                   and len(bg_masked) == len(_bg_orig_frames)
                                   else _bg_orig_frames)
                    _bt = _tj.stack([_tj.from_numpy(
                        (np.asarray(f if isinstance(f, np.ndarray) else
                                    np.asarray(f), dtype=np.float32)
                         / 127.5 - 1.0).transpose(2, 0, 1)
                    ) for f in _src_frames]).permute(1, 0, 2, 3)[None]
                    _bt = _bt.to("cuda", _vae.dtype)
                    with _tj.no_grad():
                        _enc = _vae.encode(_bt)
                        _src = (_enc.latent_dist.mode()
                                if hasattr(_enc, "latent_dist") else _enc[0])
                        _mean = _tj.tensor(
                            _vae.config.latents_mean, device=_src.device,
                            dtype=_src.dtype).view(1, -1, 1, 1, 1)
                        _std = 1.0 / _tj.tensor(
                            _vae.config.latents_std, device=_src.device,
                            dtype=_src.dtype).view(1, -1, 1, 1, 1)
                        _src = (_src - _mean) * _std
                    _lm = []
                    for _im in mk_out:
                        _a = np.asarray(_im.convert("L"), dtype=np.float32) / 255.0
                        # **LINEAR 下采样（2026-11-17 v26）**：NEAREST 会把
                        # 像素级羽化的软 alpha 量化成硬边+锯齿 → 拉回/生成的
                        # 切换变成硬边界（边缘"透明分界感"的来源）。LINEAR
                        # 保持渐变 → 每步拉回强度从人物到背景连续过渡（L3）。
                        _a = _cv2li.resize(
                            _a, (_src.shape[-1], _src.shape[-2]),
                            interpolation=_cv2li.INTER_LINEAR)
                        _lm.append(_a)
                    _li_t = _src.shape[2]
                    _lm2 = [_lm[min(i * 4, len(_lm) - 1)] for i in range(_li_t)]
                    _mnp = np.stack(_lm2).astype(np.float32)
                    # **拉回 mask 硬化（2026-11-18 v30，治半透明纱化）**：
                    # 软 alpha 直接当拉回权重时，人物内部低置信区（宽大黑裤
                    # α≈0.7~0.85）每步被 (1-α) 稀释 → 20 步累积成半透明（v29
                    # 实证：上半身 α 高实心、腿部 α 低纱化）。SD inpainting 的
                    # SetLatentNoiseMask 是硬 mask+边界羽化：二值化(α>0.5)→
                    # 腐蚀 1 格（人物内部零稀释）→ 轻羽化（边界 1 格渐变）。
                    # 软 alpha 只留给条件/合成路径（mk_out 原样）。
                    # **拉回 mask 必须"绝对包裹"（2026-11-18 v35 修鬼影）**：
                    # v30 加的 erode 是致命错误——腐蚀使遮罩内缩，原视频里
                    # **宽大黑裤/散开头发**落到 mask=0 区 → 每步被
                    # `lat*0 + src_noisy*1` 拉回原视频 → 20 步累积成双重曝光
                    # 鬼影。铁律：mask=0 区域不得含任何原人物像素。
                    # 改为 dilate 膨胀 2 格（5×5）：完全吞掉原人最宽轮廓；
                    # 原人↔新人之间的空隙区进 mask=1，交由模型按 L2 条件
                    # （telea_dim 暗化窗帘）自由生成背景。
                    _k = _cv2li.getStructuringElement(
                        _cv2li.MORPH_ELLIPSE, (5, 5))
                    _mnp = (_mnp > 0.5).astype(np.float32)
                    _mnp = _cv2li.dilate(_mnp, _k)
                    _mnp = _cv2li.GaussianBlur(_mnp, (3, 3), 0)
                    _mask_lat = _tj.from_numpy(
                        _mnp[None, None]).to(_src.device, _src.dtype)
                    _g = _tj.Generator("cpu").manual_seed(
                        int(seed) if seed >= 0 else 0)
                    _noise = _tj.randn(_src.shape, generator=_g,
                                       dtype=_tj.float32).to(_src.device,
                                                             _src.dtype)
                    _li_ctx.update({"src": _src, "mask": _mask_lat,
                                    "noise": _noise})
                    print(f"[wan][latent-inpaint] armed src={tuple(_src.shape)} "
                          f"mask人物={float(_mask_lat.mean())*100:.1f}%",
                          flush=True)
            except Exception as _eli:  # noqa: BLE001
                print(f"[wan][latent-inpaint] 准备失败（{type(_eli).__name__}: "
                      f"{_eli}），本次关闭", flush=True)
                _li_ctx.clear()


        kwargs = dict(image=ref, pose_video=pose, face_video=face,
                      prompt=prompt, negative_prompt=negative_prompt or None,
                      height=h, width=w,
                      segment_frame_length=int(segment_frame_length),
                      num_inference_steps=int(steps),
                      guidance_scale=float(guidance_scale),
                      mode=_mode, generator=gen, output_type="np",
                      **extra)
        if _pb is not None:
            # pipeline 禁止同时给 prompt 与 prompt_embeds（Cannot forward both）
            kwargs["prompt"] = None
            kwargs["negative_prompt"] = None
            kwargs["prompt_embeds"] = _pb
            if _nb is not None:
                kwargs["negative_prompt_embeds"] = _nb
        # **callback 仅用于进度上报/NaN 探测，可整体关闭（2026-11-16）**：
        # 官方最小脚本（不带 callback）在 3090 上能正常出片，带 callback
        # 的调用在同样环境出雪花 → callback 机制是唯一剩余差异，故支持
        # WAN_CALLBACK=0 完全绕过（不传任何 callback 参数，与官方示例一致）。
        # **latent inpainting 必须走 callback**（2026-11-17）：supervisor 里
        # 曾为规避雪花设 WAN_CALLBACK=0；但 latent-inpaint 的每步拉回依赖
        # 官方 callback_on_step_end，故 **开启 latent_inpaint 时强制启用**，
        # 其他情况仍尊重 WAN_CALLBACK 开关。
        _need_cb = bool(_li_ctx) or os.environ.get("WAN_CALLBACK", "1") != "0"
        if _need_cb:
            kwargs["callback_on_step_end"] = _nan_cb
            kwargs["callback_on_step_end_tensor_inputs"] = ["latents"]
            print(f"[wan] callback 已启用（latent_inpaint="
              f"{bool(_li_ctx)}）pipe={type(pipe).__name__}", flush=True)
        print(f"[wan] animate2(WanAnimatePipeline/{_mode}) {w}x{h} "
              f"pose={len(pose)}帧 seg={segment_frame_length} "
              f"steps={steps} seed={seed}", flush=True)
        t0 = time.time()
        out = pipe(**kwargs)
        frames_np = out.frames[0]
        # 诊断（真机排障用）：确认 pipeline 输出形状/数值范围是否如预期
        _nan = bool(np.isnan(np.asarray(frames_np)).any())
        print(f"[wan] frames_np shape={getattr(frames_np, 'shape', '?')} "
              f"dtype={getattr(frames_np, 'dtype', '?')} "
              f"min={float(np.min(frames_np)):.3f} "
              f"max={float(np.max(frames_np)):.3f} "
              f"mean={float(np.mean(frames_np)):.3f}", flush=True)
        # diffusers 的 WanPipeline 在 output_type="np" 下返回
        # (num_frames, height, width, channels)，范围 [0, 1]。若返回 [-1,1]
        # 或通道在前，图像会错乱（真机：黄黑色块/横向条带），故按形状自适应。
        if _nan:
            raise RuntimeError(
                "wan.animate2: pipeline 输出含 NaN（全黑帧）——疑似 group "
                "offloading 搬运竞争或 scheduler 数值不稳定，已快速失败，"
                "不要保存黑视频")
        arr = np.asarray(frames_np)
        if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))      # (T,C,H,W) → (T,H,W,C)
        if arr.min() < -0.05:
            arr = (arr + 1.0) / 2.0                     # [-1,1] → [0,1]
        frames = [_Image.fromarray(
            (np.clip(f, 0, 1) * 255).round().astype("uint8"))
            for f in arr]
        # **composite 像素级合成（2026-11-15 暗晕根治疗法）**：diffusers
        # replace 无语义合成，黑洞抹除 + mask nearest 下采样使人物边缘一圈
        # 暗色光晕（dilate 越大越重）。输出端自行合成：softmask 外扩
        # composite_expand px（混合带推到 mask 外的干净背景区，避免原人物
        # 残影）+ 羽化 → final = gen×soft + 原始背景×(1-soft)。
        print(f"[wan] composite 参数: fill_black={composite_fill_black:g} "
              f"dehaze={composite_dehaze} expand={composite_expand} "
              f"feather={composite_feather} gain={composite_gain:g} "
              f"inpaint_expand={inpaint_expand} "
              f"edge_clean={composite_edge_clean}", flush=True)
        if composite in (True, "1", "true", "True", 1) and mk_out is not None:
            import cv2 as _cv2
            _ce = int(composite_expand); _cf = int(composite_feather)
            _soft = []
            for _m_pil in mk_out:
                # **软 alpha 兼容（2026-11-17 trimap）**：video-person-mask 开了
                # mask_matting 时输出的是**连续 alpha**（0~255 渐变，发丝级边缘）。
                # 这里若仍然 `>0.5` 二值化，软边会被重新砍成硬边，trimap 白做。
                # 判定：如果该帧灰度同时存在 0 和 255 之间的中间值（>2%），
                # 视为软 alpha 直接保留；否则按原二值流程走（兼容旧行为）。
                _raw = np.asarray(_m_pil, dtype=np.float32) / 255.0
                _mid = ((_raw > 0.05) & (_raw < 0.95)).mean()
                if _mid > 0.02:
                    _m = np.clip(_raw, 0.0, 1.0)
                    _is_soft = True
                else:
                    _m = (_raw > 0.5).astype(np.uint8)
                    _is_soft = False
                # **负值 = 腐蚀收缩（2026-11-16 用户方案 2）**：SAM3 轮廓与
                # 新角色轮廓不重合时，mask 边界外一圈被模型画暗；把合成边界
                # **内缩**到人物干净内部即可完全避开污染带（比颜色修补可靠）。
                if _is_soft:
                    # 软 alpha：不做形态学外扩（会破坏发丝级渐变），
                    # 只用 composite_expand 的方向感做轻度偏移
                    if _ce != 0:
                        print(f"[wan] 软 alpha 输入：跳过 expand={_ce} 的"
                              f"形态学处理（保留 trimap 软边）", flush=True)
                elif _ce > 0:
                    _m = _cv2.dilate(_m, np.ones((_ce * 2 + 1,) * 2, np.uint8))
                elif _ce < 0:
                    _m = _cv2.erode(_m, np.ones((-2 * _ce + 1,) * 2, np.uint8))
                _sv = _m.astype(np.float32)
                if _cf > 0:
                    _sv = _cv2.GaussianBlur(_sv, (0, 0), _cf / 2.0)
                    _sv = np.clip(_sv, 0.0, 1.0)
                _soft.append(_sv[..., None])
            _soft = np.stack(_soft)  # [T,H,W,1]
            _genf = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
            _bgf = np.stack(_bg_orig_frames).astype(np.float32)
            # **混合带亮度对齐（2026-11-16 第三层暗带治法）**：feather 带
            # 内 gen 边缘偏暗 × 背景亮 → 加权平均出 ~15px 浅暗带。按帧统计
            # 带内两侧灰度比 g（限幅 composite_gain），gain_map = 1+(g-1)*
            # (1-soft)：本体(soft=1)不变、贴 bg 侧渐入 g 倍，亮度单调无断层。
            _cg = float(composite_gain or 0)
            if _cg > 0:
                _gmax = min(_cg, 1.35)
                _gs = []
                for _t in range(_genf.shape[0]):
                    _s1 = _soft[_t, ..., 0]
                    _band = (_s1 > 0.15) & (_s1 < 0.85)
                    if _band.sum() > 500:
                        _gg = _genf[_t].mean(-1)
                        _bb = _bgf[_t].mean(-1)
                        _mg = _gg[_band].mean()
                        _mb = _bb[_band].mean()
                        _g = 1.0 if _mg < 5 else _mb / _mg
                        # 双向增益（2026-11-16 阴影修复）：亮人物⊕暗背景时
                        # _mb/_mg<1，原单向钳到1.0不补偿 → 羽化带留深色描边。
                        # 放开下限，允许压暗人物边缘向背景靠拢；上下限对称
                        # 1/_gmax~_gmax，避免过曝/死黑。
                        _g = float(min(max(_g, 1.0 / _gmax), _gmax))
                    else:
                        _g = 1.0
                    _gs.append(_g)
                _gs = np.asarray(_gs, dtype=np.float32)[:, None, None, None]
                _gain_map = 1.0 + (_gs - 1.0) * (1.0 - _soft)
                _genf = _genf * _gain_map
                print(f"[wan] composite 增益对齐: g∈[{_gs.min():.2f},"
                      f"{_gs.max():.2f}]（混合带亮度台阶补偿）", flush=True)
            # **边缘反预乘（2026-11-17，用户方案③的严格形式）**
            # 模型在 mask 边界生成的是「半透明前景叠在黑抹除层上」的结果：
            #   I = a·F + (1-a)·0    （B 被抹成纯黑）
            # 直接合成 `I·a + bg·(1-a)` 会把已经衰减过的 I 再乘一次 a ⇒ 边界
            # 系统性偏暗（这就是黑边的数学成因）。正解先反预乘还原 F = I/a：
            #   F = I / max(a, a_min)
            # 再正常合成 `F·a + bg·(1-a)` = I + bg·(1-a) ⇒ 边界亮度回到背景水平。
            # a_min 限制除法增益（a→0 时噪声会被放大），只在过渡带生效。
            _up = float(composite_unpremul or 0)
            if _up > 0:
                # _soft 已经是 (T,H,W,1)，别再升维（否则与 (T,H,W,3) 广播失败）
                _a = _soft
                _a_c = np.clip(_a, 1.0 / max(_up, 1.0), 1.0)
                _band3 = (_a[..., 0] > 0.02) & (_a[..., 0] < 0.98)
                _f = np.where(_band3[..., None],
                              np.clip(_genf / _a_c, 0, 255), _genf)
                print(f"[wan] 边缘反预乘: gain≤{_up:g}x，"
                      f"过渡带 {int(_band3.sum())} 像素", flush=True)
                _genf = _f
            _mixed = np.clip(_genf * _soft + _bgf * (1.0 - _soft), 0, 255)
            # **暗晕抑制（2026-11-16「人物一圈黑影」根治）**：模型在黑抹除
            # 区生成时，人物外轮廓天然带一圈比背景暗的边缘；composite 的
            # gain 只覆盖 0.15<s<0.85 的过渡带，贴着人物那圈（s→1）压不到。
            # 这里在**混合带内**挑出"明显比背景暗"的像素，往背景亮度提拉
            # （只提不压，且限幅），实心人物本体（s>0.97）不动。
            if composite_dehaze in (True, "1", "true", "True", 1):
                _s0 = _soft[..., 0]
                _lum_g = _mixed.mean(-1)
                _lum_b = _bgf.mean(-1)
                _band2 = (_s0 > 0.03) & (_s0 < 0.97)
                _dark = _band2 & (_lum_g < _lum_b * 0.82)
                if _dark.any():
                    _target = _bgf * float(min(max(composite_dehaze_strength,
                                                   0.0), 1.0))
                    _mixed = np.where(_dark[..., None],
                                      np.maximum(_mixed, _target), _mixed)
                    print(f"[wan] 暗晕抑制: {int(_dark.sum())} 像素 "
                          f"({100.0*_dark.sum()/_band2.size:.2f}%)", flush=True)
            # **mask 内近黑空白回填（2026-11-16 硬黑边根治）**：模型只在
            # mask 内重绘，若新角色体型/服装与原 SAM3 分割的人物不一致
            # （绿裙比黑裙宽/窄），mask 内就会剩一圈**模型画成近黑色**的空区；
            # composite 按 mask 贴回时这圈黑边原样落地 → 人物像剪纸贴上去。
            # 这里把 mask 内**近黑**（RGB 三通道都很低）的像素换成原始背景：
            # 深色衣物通常 >30，纯黑空白 <20，阈值可区分。
            _bf = float(composite_fill_black or 0)
            if _bf > 0:
                _d0 = _soft[..., 0]
                _dkm = (_genf.max(-1) < _bf) & (_d0 > 0.25)
                # **只在 mask 边界带回填（2026-11-16 真机）**：纯阈值回填会
                # 误伤人物**内部**的深色区域（黑发、深色衣物）——实测阈值
                # 提到 55 时头顶被填成白斑。黑边残渣只出现在 mask **边界**
                # 附近（原片人物轮廓与新角色轮廓不重合处），故限制在
                # mask 腐蚀 30px 的环带内，内部深色一律不动。
                try:
                    import cv2 as _cv2b
                    # _d0 可能是 (H,W) 或 (H,W,1)/(H,W,3) → 先压成 2D 且连续
                    # （cv2.erode 只吃 CV_8UC1/CV_8UC3，非连续/多余维度会抛
                    #  Errno "error"）
                    _vis = np.asarray(_d0)
                    if _vis.ndim == 3:
                        _vis = _vis[..., 0]
                    _vis = np.ascontiguousarray(
                        (_vis > 0.5).astype(np.uint8))
                    _k = np.ones((31, 31), np.uint8)
                    _er = _cv2b.erode(_vis, _k)
                    _band = (_vis.astype(bool) & ~_er.astype(bool))
                    if _dkm.ndim == 3:
                        _band = _band[..., None]
                    _n_before = int(_dkm.sum())
                    _dkm = _dkm & _band
                    print(f"[wan] 黑边回填限界: {_n_before} -> "
                          f"{int(_dkm.sum())} 像素（仅边界带）", flush=True)
                except Exception as _eb:  # noqa: BLE001
                    print(f"[wan] 边界带限界跳过（{type(_eb).__name__}: "
                          f"{_eb}）", flush=True)
                if _dkm.any():
                    _mixed = np.where(_dkm[..., None], _bgf, _mixed)
                    print(f"[wan] 黑边回填: {int(_dkm.sum())} 像素 "
                          f"({100.0*_dkm.sum()/_d0.size:.2f}%) 阈值<{_bf:g}",
                          flush=True)
            # **边缘颜色去污（2026-11-16，用户方案落地）**：黑边的真正成因是
            # **边缘像素颜色被污染** —— 抹黑（bg×(1-mask)）后模型在 mask 边界
            # 把人物画暗（保留黑洞的暗调），合成时 softmask 在边界是半透明，
            # 半暗的前景叠在原背景上 ⇒ 肉眼可见的一圈黑边/黑雾。
            # 按 alpha 合成公式 F = (I - (1-a)B)/a 的思路，工程上用
            # **人物内部颜色重建边界带**：
            #   1) core  = erode(mask, d)          —— 干净内部（未被抹黑影响）
            #   2) band  = mask - core             —— 污染环带
            #   3) inpaint(gen, band)              —— 用内部/周围像素填补环带
            #   4) 只在 band 内替换                     —— 内部细节一律不动
            # 相比 fill_black（按亮度阈值猜哪些是黑边，会误伤黑发/黑手套），
            # 这是**几何确定性**的：环带内的暗调一律由内部亮色外推。
            _ec = int(composite_edge_clean or 0)
            if _ec > 0:
                try:
                    import cv2 as _cv2c
                    _kb = np.ones((_ec * 2 + 1, _ec * 2 + 1), np.uint8)
                    _tot_band = _tot_fix = 0
                    # **必须逐帧**：_soft/_genf 是 (T,H,W,·) 全帧数组，
                    # cv2.erode/inpaint 只吃 2D 单图（把 3D 当多通道会
                    # 把整幅腐蚀成空 → 之前日志"core/band 为空"）。
                    for _t in range(_soft.shape[0]):
                        _m8 = np.ascontiguousarray(
                            (_soft[_t, ..., 0] > 0.5).astype(np.uint8))
                        if not _m8.any():
                            continue
                        _core = _cv2c.erode(_m8, _kb)
                        _band = (_m8.astype(bool) & ~_core.astype(bool))
                        if not (_band.any() and _core.any()):
                            continue
                        _gf8 = np.clip(_genf[_t], 0, 255).astype(np.uint8)
                        _fixed = _cv2c.inpaint(
                            _gf8, (_band.astype(np.uint8)) * 255, 7,
                            _cv2c.INPAINT_TELEA)
                        # **只修"暗"的边缘**：亮边（白衣服）本来就干净，
                        # 强行 inpaint 会抹掉真实布料边缘细节。
                        _lum_g = _gf8.astype(np.float32).mean(-1)
                        _lum_f = _fixed.astype(np.float32).mean(-1)
                        _need = _band & (_lum_f > _lum_g + 2.0)
                        _tot_band += int(_band.sum())
                        if _need.any():
                            _tot_fix += int(_need.sum())
                            _mixed[_t] = np.where(
                                _need[..., None],
                                _fixed.astype(np.float32), _mixed[_t])
                    print(f"[wan] 边缘去污: 环带 {_ec}px 累计 {_tot_band} 像素，"
                          f"实际修正 {_tot_fix}（仅补亮）", flush=True)
                except Exception as _eec:  # noqa: BLE001
                    print(f"[wan] 边缘去污失败（{type(_eec).__name__}: "
                          f"{_eec}）", flush=True)
            _mixed = np.clip(_mixed, 0, 255)
            frames = [_Image.fromarray(f.astype("uint8")) for f in _mixed]
            print(f"[wan] composite 合成: expand={_ce} feather={_cf}"
                  f"（原始背景混合）", flush=True)
        print(f"[wan] gen done {len(frames)}帧 {fps}fps "
              f"{time.time()-t0:.1f}s", flush=True)
        return {"video": {"frames": frames, "fps": fps,
                          "num_frames": len(frames)},
                "seed": seed, "fps": fps}

    @registry.register(
        "wan.animate",
        inputs={"model": "MODEL", "image": "IMAGE",
                "pose_video": "VIDEO", "face_video": "VIDEO",
                "background_video": "VIDEO", "mask_video": "VIDEO",
                "prompt": "STRING", "negative_prompt": "STRING",
                "mode": "STRING", "segment_frame_length": "INT",
                "prev_segment_conditioning_frames": "INT",
                "motion_encode_batch_size": "INT",
                "height": "INT", "width": "INT", "steps": "INT",
                "guidance_scale": "FLOAT", "seed": "INT"},
        outputs={"video": "VIDEO", "seed": "INT", "fps": "INT"},
        description="Wan-Animate：mode=animate 让角色图按参考视频动作/表情动起来；"
                    "mode=replace 把背景视频里的人换成新角色（需 background_"
                    "video + mask_video：白=生成/黑=保留）。pose_video 与 "
                    "face_video 必须是**预处理后的**骨骼/人脸表示（官方 "
                    "Wan2.2 仓库 preprocess 脚本产出），不能直接喂原始视频。")
    def _op_animate(model, image=None, pose_video=None, face_video=None,
                    background_video=None, mask_video=None,
                    prompt="", negative_prompt="", mode="animate",
                    segment_frame_length=77,
                    prev_segment_conditioning_frames=1,
                    motion_encode_batch_size=0,
                    height=0, width=0, steps=20,
                    guidance_scale=1.0, seed=-1):
        import random

        import numpy as np
        import torch
        from PIL import Image as _Image

        pipe = model
        mode = (mode or "animate").lower()
        if mode not in ("animate", "replace"):
            raise ValueError("wan.animate: mode 须为 animate / replace")
        if image is None:
            raise ValueError("wan.animate: image（角色参考图）必填")
        if mode == "replace" and (background_video is None
                                  or mask_video is None):
            raise ValueError("wan.animate: replace 模式需 background_video "
                             "与 mask_video（白=生成区/黑=保留区）")

        def _pils(obj):
            if obj is None:
                return None
            if isinstance(obj, _Image.Image):
                return [obj]
            if isinstance(obj, dict):
                if obj.get("frames"):
                    return list(obj["frames"])
                if obj.get("path"):
                    return obj["path"]  # diffusers load_video 支持路径
            return obj

        def _frame_array(obj):
            """VIDEO → (N,H,W,3) float[0,1] ndarray（diffusers 期望形态）。"""
            pils = _pils(obj)
            if isinstance(pils, str):
                from diffusers.utils import load_video
                arr = load_video(pils)
                return arr
            if isinstance(pils, list) and pils and isinstance(
                    pils[0], _Image.Image):
                # 统一成 float32 [0,1]：diffusers 各 pipe 对 uint8/float 的
                # 容忍度不一致，显式归一化最稳（PIL 输入内部也是这么转的）
                return [np.asarray(f.convert("RGB"), dtype=np.float32) / 255.0
                        for f in pils]
            return pils

        ref = _pils(image)
        ref = ref[0] if isinstance(ref, list) else ref
        pose = _frame_array(pose_video)
        face = _frame_array(face_video)
        bg = _frame_array(background_video) if background_video is not None else None
        msk = _frame_array(mask_video) if mask_video is not None else None

        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        h, w = int(height), int(width)
        if h <= 0 or w <= 0:
            w, h = ref.size
        mod = 16
        h = max(mod, round(h / mod) * mod)
        w = max(mod, round(w / mod) * mod)

        fps = int(getattr(pipe, "_wan_fps", 30))
        kwargs = dict(image=ref, pose_video=pose, face_video=face,
                      prompt=prompt, negative_prompt=negative_prompt or None,
                      height=h, width=w,
                      segment_frame_length=int(segment_frame_length),
                      guidance_scale=float(guidance_scale),
                      num_inference_steps=int(steps),
                      mode=mode, generator=gen, output_type="np")
        if int(prev_segment_conditioning_frames) >= 0:
            # 段间时序引导（diffusers 默认 1；5 更连贯但更吃显存）
            kwargs["prev_segment_conditioning_frames"] = int(
                prev_segment_conditioning_frames)
        if int(motion_encode_batch_size) > 0:
            # 动作 VAE encode 的 batch（大分辨率/长片段时调低可省显存）
            kwargs["motion_encode_batch_size"] = int(motion_encode_batch_size)
        if mode == "replace":
            kwargs["background_video"] = bg
            kwargs["mask_video"] = msk
        t0 = time.time()
        out = pipe(**kwargs)
        frames_np = out.frames[0]
        # 与 wan.gen 同款自适应：diffusers 的 output_type="np" 一般给
        # (T,H,W,C)/[0,1]，但版本间可能给 (T,C,H,W) 或 [-1,1]，
        # 不处理会得到横向条带/全黑（真机在 wan.gen 上踩过）。
        arr = np.asarray(frames_np)
        print(f"[wan] animate frames shape={arr.shape} "
              f"min={float(arr.min()):.3f} max={float(arr.max()):.3f}",
              flush=True)
        if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.min() < -0.05:
            arr = (arr + 1.0) / 2.0
        frames = [_Image.fromarray(
            (np.clip(f, 0, 1) * 255).round().astype("uint8")) for f in arr]
        print(f"[wan] animate({mode}) done {len(frames)}帧 {fps}fps "
              f"{time.time()-t0:.1f}s", flush=True)
        return {"video": {"frames": frames, "fps": fps,
                          "num_frames": len(frames)},
                "seed": seed, "fps": fps}


    @registry.register(
        "wan.vace",
        inputs={"model": "MODEL", "prompt": "STRING",
                "negative_prompt": "STRING", "video": "VIDEO",
                "mask": "VIDEO", "reference_image": "IMAGE",
                "conditioning_scale": "FLOAT", "num_frames": "INT",
                "height": "INT", "width": "INT", "steps": "INT",
                "guidance_scale": "FLOAT", "guidance_scale_2": "FLOAT",
                "seed": "INT"},
        outputs={"video": "VIDEO", "seed": "INT", "fps": "INT"},
        description="Wan-VACE 可控生成（**人物替换首选**）：给定原视频 video + "
                    "遮罩 mask（白=重绘区/黑=保留区）+ 新角色 reference_image，"
                    "在原片动作/构图上重绘遮罩区。相比 Wan-Animate **不需要 "
                    "pose/face 预处理**，可直接用 video-person-mask 的产物。"
                    "conditioning_scale 控制条件强度（1.0 为训练默认）。")
    def _op_vace(model, prompt="", negative_prompt="", video=None, mask=None,
                 reference_image=None, conditioning_scale=1.0,
                 num_frames=0, height=0, width=0, steps=30,
                 guidance_scale=5.0, guidance_scale_2=0.0, seed=-1):
        import random

        import numpy as np
        import torch
        from PIL import Image

        pipe = model
        if video is None:
            raise ValueError("wan.vace: video（原视频）必填")
        if not prompt:
            raise ValueError("wan.vace: prompt 必填（描述替换后的画面）")

        def _frames(obj):
            """VIDEO 对象 → [PIL] 或路径（diffusers 期望 list[PIL|ndarray]）。"""
            if obj is None:
                return None
            if isinstance(obj, dict):
                fr = obj.get("frames")
                if fr:
                    return list(fr)
                p = obj.get("path")
                if p:
                    from diffusers.utils import load_video
                    return load_video(p)          # → ndarray 列表
            return obj

        vid = _frames(video)
        msk = _frames(mask)
        # **video/mask 帧数对齐**（真机：video.to_frames 与 video.person_mask
        # 的抽帧口径差 1 帧 → "Length of video 120 and mask 121 do not match"）。
        # 取两者较短的长度，再向下对齐 4k+1。
        if msk is not None and hasattr(vid, "__len__") and hasattr(msk, "__len__"):
            n0 = min(len(vid), len(msk))
            n_al = _align_frames(n0)
            if n_al > n0:                      # 对齐后不能超
                n_al = max(5, ((n0 - 1) // 4) * 4 + 1)
            if len(vid) != n_al:
                vid = vid[:n_al]
            if len(msk) != n_al:
                msk = msk[:n_al]
            print(f"[wan] vace 对齐 video/mask -> {n_al} 帧"
                  f"（原 {len(vid) if n_al == len(vid) else n0}）", flush=True)
        ref = reference_image
        if isinstance(ref, dict):
            fr = ref.get("frames")
            ref = (fr[0] if isinstance(fr, list) and fr else
                   Image.open(ref["path"]).convert("RGB")
                   if ref.get("path") else None)

        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        # 帧数/尺寸：未给则跟随原视频
        n = int(num_frames)
        if n <= 0:
            n = len(vid) if hasattr(vid, "__len__") else 81
        n = _align_frames(n)
        h, w = int(height), int(width)
        if h <= 0 or w <= 0:
            # 未指定尺寸：**按源宽高比 + 480p 像素预算**算，不能直接取源分辨率
            # （1344×768 的源直传会在 24GB 卡上 OOM——真机记录：124帧 832×480 已
            # 经 22.9GB 峰值）。
            first = vid[0] if hasattr(vid, "__getitem__") else None
            if first is not None:
                if isinstance(first, Image.Image):
                    sw, sh = first.size
                else:
                    a = np.asarray(first)
                    sh, sw = (a.shape[0], a.shape[1]) if a.ndim >= 2 else (480, 832)
                budget = 832 * 480
                s = (budget / max(1, sw * sh)) ** 0.5
                w = max(64, round(sw * s / 16) * 16)
                h = max(64, round(sh * s / 16) * 16)
            else:
                w, h = 832, 480
        mod = 16
        h = max(mod, round(h / mod) * mod)
        w = max(mod, round(w / mod) * mod)

        # ---- 关键修复（真机：画面底部一条环绕带 + 尾部纯色帧）----
        # WanVACEPipeline.preprocess_conditions 会按「等比缩到不超过 height*width
        # 的面积」+「向下对齐 vae_scale_factor_spatial × patch_size[1]」**重新**算
        # video 尺寸，而 latents 用的是调用方传的 height/width **原样**。
        # 两者不等 ⇒ cond latent 与 noise latent 空间维对不上：
        #   1344×768 源 + 320×512 目标 ⇒ pipeline 缩成 512×288 ⇒ cond 36 行
        #   vs noise 40 行 ⇒ transformer 拼接条件后尾部被 padding ⇒ 解码出
        #   “滚动带”（H 方向错位）+ 尾部纯色帧。
        # 做法：先把帧**精确** resize 到 (w,h)。此时 pipeline 内部
        # `video_h*video_w > height*width` 为 False（相等）⇒ 不再缩放 ⇒ 尺寸一致。
        def _cover_crop(im, tw, th):
            """等比缩放 + 居中裁剪（不拉伸变形）。"""
            sw, sh = im.size
            s = max(tw / sw, th / sh)
            nw, nh = max(tw, round(sw * s)), max(th, round(sh * s))
            im = im.resize((nw, nh), Image.LANCZOS)
            x, y = (nw - tw) // 2, (nh - th) // 2
            return im.crop((x, y, x + tw, y + th))

        def _norm_frames(frames, tw, th):
            out = []
            for f in frames:
                im = f if isinstance(f, Image.Image) else Image.fromarray(
                    np.asarray(f).astype("uint8"))
                if im.mode != "RGB":
                    im = im.convert("RGB")
                if im.size != (tw, th):
                    im = _cover_crop(im, tw, th)
                out.append(im)
            return out

        if hasattr(vid, "__len__") and len(vid):
            first = vid[0]
            first_sz = (first.size if isinstance(first, Image.Image)
                        else tuple(np.asarray(first).shape[1::-1]))
            vid = _norm_frames(vid, w, h)
            print(f"[wan] vace 帧统一 {first_sz} -> {w}x{h}（{len(vid)} 帧）",
                  flush=True)
        if msk is not None and hasattr(msk, "__len__"):
            msk = _norm_frames(msk, w, h)
        # 帧数：4k+1 对齐且**不超源**（超了会让 pipeline 自己 pad，尾部就出脏帧）
        if hasattr(vid, "__len__") and len(vid) < n:
            n_fix = max(5, ((len(vid) - 1) // 4) * 4 + 1)
            print(f"[wan] vace 帧数不足（源 {len(vid)} < {n}）-> n={n_fix}",
                  flush=True)
            n = n_fix
        if hasattr(vid, "__getitem__"):
            vid = vid[:n]
        if msk is not None and hasattr(msk, "__getitem__"):
            msk = msk[:n]
        print(f"[wan] vace 条件帧数 n={n} "
              f"mask={'yes' if msk is not None else 'no'}", flush=True)

        fps = int(getattr(pipe, "_wan_fps", WAN_FPS_DEFAULT))
        kwargs = dict(prompt=prompt, negative_prompt=negative_prompt or None,
                      video=vid, mask=msk,
                      conditioning_scale=float(conditioning_scale),
                      height=h, width=w, num_frames=n,
                      num_inference_steps=int(steps),
                      guidance_scale=float(guidance_scale),
                      generator=gen, output_type="np")
        if ref is not None:
            kwargs["reference_images"] = [ref]
        if guidance_scale_2 and float(guidance_scale_2) > 0:
            kwargs["guidance_scale_2"] = float(guidance_scale_2)

        # **VACE 干净 pipeline 重建（2026-11-16 真机决定性对照）**：
        # 同一份 1.3B 权重，**diffusers 官方脚本**（直接
        # `WanVACEPipeline(...) + pipe.to("cuda")`）出片完全正常；
        # 而经 `wan.pipe` 装配的 pipeline 无论怎么调参都是彩色噪声。
        # Animate 当初也是同一现象（独立装配解决）。故这里用组件重建一个
        # **不带任何 offload hook / 哨兵 / 缓存状态**的 pipeline，等价官方路径。
        if _is_vace_pipe(pipe) and not getattr(pipe, "_flowx_clean", False):
            try:
                from diffusers import WanVACEPipeline as _WVP
                _tf = pipe.transformer
                for _m in (getattr(pipe, "transformer", None),
                           getattr(pipe, "text_encoder", None),
                           getattr(pipe, "vae", None)):
                    if _m is not None and hasattr(_m, "_hf_hook"):
                        try:
                            del _m._hf_hook
                        except Exception:  # noqa: BLE001
                            pass
                _clean = _WVP(
                    tokenizer=pipe.tokenizer,
                    text_encoder=pipe.text_encoder,
                    vae=pipe.vae,
                    transformer=_tf,
                    scheduler=pipe.scheduler,
                )
                # 组件已在卡上时**不要再 to("cuda")**（会复制一份 → 24G
                # 卡上 1.3B+TE 共 18G 直接 OOM）
                _need = [m for m in (_clean.transformer, _clean.text_encoder,
                                     _clean.vae)
                         if m is not None
                         and getattr(m, "device", None) is not None
                         and m.device.type != "cuda"]
                if _need:
                    for _m in _need:
                        _m.to("cuda")
                    print(f"[wan] vace 重建后搬到 GPU 的组件 {len(_need)} 个",
                          flush=True)
                else:
                    print("[wan] vace 组件已全在 GPU，跳过搬运", flush=True)
                _clean._flowx_clean = True
                pipe = _clean
                print("[wan] vace 干净 pipeline 重建完成（官方路径，无 hook）",
                      flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[wan] vace 重建失败，沿用原 pipe："
                      f"{type(_e).__name__}: {_e}", flush=True)

        t0 = time.time()
        out = pipe(**kwargs)
        arr = np.asarray(out.frames[0])
        print(f"[wan] vace frames shape={arr.shape} "
              f"min={float(arr.min()):.3f} max={float(arr.max()):.3f}",
              flush=True)
        if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.min() < -0.05:
            arr = (arr + 1.0) / 2.0
        frames = [Image.fromarray((np.clip(f, 0, 1) * 255).round().astype("uint8"))
                  for f in arr]
        print(f"[wan] vace done {len(frames)}帧 {fps}fps {time.time()-t0:.1f}s",
              flush=True)
        return {"video": {"frames": frames, "fps": fps,
                          "num_frames": len(frames)},
                "seed": seed, "fps": fps}



def _apply_lora_config(pipe, cfg):
    """解析 wan.lora_load 的配置串并应用。

    格式：`LORA=/abs/a.safetensors:0.8,/abs/b.safetensors:1|RELIGHT=/abs/r.safetensors:1`
    - LORA 走 diffusers 的 load_lora_weights（运行时 hook）
    - RELIGHT 走 _merge_relight_lora（直接合并进权重，必须在挂 offload 前）
    两者失败都只告警不中断——打光/加速 LoRA 属于增强项，不该阻断出片。
    """
    for part in str(cfg or "").split("|"):
        if "=" not in part:
            continue
        tag, _, body = part.partition("=")
        items = [x for x in body.split(",") if x.strip()]
        if not items:
            continue
        if tag == "RELIGHT":
            for it in items:
                path, _, sc = it.rpartition(":")
                try:
                    scale = float(sc)
                except ValueError:
                    path, scale = it, 1.0
                try:
                    _merge_relight_lora(pipe.transformer, path, scale)
                except Exception as e:      # noqa: BLE001
                    print(f"[wan] ⚠ relight lora 合并失败 "
                          f"({type(e).__name__}: {str(e)[:90]})", flush=True)
        elif tag == "LORA":
            try:
                pipe = _load_lora(pipe, ",".join(items), 1.0)
                print(f"[wan] lora 已挂载 {len(items)} 项", flush=True)
            except Exception as e:          # noqa: BLE001
                print(f"[wan] ⚠ lora 挂载失败 "
                      f"({type(e).__name__}: {str(e)[:90]})", flush=True)
    return pipe




def _build_animate2_int8(transformer_dir, int8_path, dtype,
                         map_json=None):
    """Wan-Animate transformer 的 **int8 混合加载**（2026-11-16）。

    为什么：bf16 权重 34.5GB 全在内存（CPU 常驻 + 逐块上卡），冷启动要
    从盘读 33GB（120MB/s ≈ 5 分钟），而且 24GB 卡上再分不出显存给更高
    分辨率。共享盘的 Comfy-Org `wan2.2_animate_14B_int8_convrot` 只有
    17.1GB，量化层（480 个 Linear）用 int8 常驻，其余（norm/bias/face_adapter
    等）仍用 bf16 —— 实测内存 33GB → 21GB，读盘量减 40%。

    实现要点：
      1. meta 构造模型（零内存空壳）
      2. **只**从 diffusers bf16 分片读非量化键（bias/norm/buffer/face_adapter）
      3. 量化层从 int8 文件读 (weight_int8, weight_scale) → 换成
         `_Int8DequantLinear`（含 convrot 逆旋转，H3 已真机验证的语义）
      4. 键名映射表由 build_map.py 离线生成（ComfyUI → diffusers）
    """
    import json as _json
    import torch as _t
    from diffusers import WanAnimateTransformer3DModel as _TR
    from safetensors import safe_open as _safe_open
    from safetensors.torch import load_file as _lf

    t0 = time.time()
    cfg = _json.load(open(os.path.join(transformer_dir, "config.json")))
    with _t.device("meta"):
        model = _TR.from_config(cfg, torch_dtype=dtype)

    map_json = map_json or os.environ.get(
        "WAN_INT8_MAP", "/var/lib/flowx-inference/models/wan-animate-diffusers/"
                        "comfy2diff_wananimate.json")
    mapping = _json.load(open(map_json))
    dk = set(model.state_dict().keys())

    # --- 量化层清单：映射后是 *.weight 且有对应 .weight_scale ---
    quant_src, quant_dst = {}, {}
    with _safe_open(int8_path, framework="pt") as f:
        src_keys = set(f.keys())
        for ck, cd in mapping.items():
            if not ck.endswith(".weight"):
                continue
            sk = ck[: -len(".weight")] + ".weight_scale"
            if sk in src_keys and cd in dk:
                quant_src[cd] = (ck, sk)
    print(f"[wan][int8] 量化层 {len(quant_src)} 个", flush=True)

    # --- 非量化键：从 diffusers bf16 分片读（跳过量化层）---
    idx = _json.load(open(os.path.join(
        transformer_dir, "diffusion_pytorch_model.safetensors.index.json")))
    need = {k for k in dk if k not in quant_src}
    got = {}
    for shard in sorted(set(idx["weight_map"].values())):
        part = _lf(os.path.join(transformer_dir, shard))
        for k, v in part.items():
            if k in need:
                got[k] = v
        del part
    miss = need - set(got)
    print(f"[wan][int8] 非量化键 {len(got)}/{len(need)} 已读"
          f"（缺 {len(miss)}）{time.time()-t0:.1f}s", flush=True)
    if miss:
        print(f"[wan][int8] ⚠ 缺失样例 {sorted(miss)[:5]}", flush=True)
    model.load_state_dict(got, strict=False, assign=True)
    del got

    # --- 替换量化层 ---
    convrot_default = int(os.environ.get("WAN_INT8_CONVROT", "256"))
    with _safe_open(int8_path, framework="pt") as f:
        for cd, (wk, sk) in quant_src.items():
            w = f.get_tensor(wk)
            s = f.get_tensor(sk)
            if w.dtype != _t.int8:
                continue
            # **每层 groupsize 必须从 .comfy_quant metadata 读**（不全是 256，
            # adaln 一类是 64）：H3 真机教训——用错 groupsize 做 Hadamard
            # 逆旋转 = 权重基变换错，整模型输出花屏。
            g = convrot_default
            cqk = wk[: -len(".weight")] + ".comfy_quant"
            if cqk in src_keys:
                try:
                    _md = _json.loads(f.get_tensor(cqk).numpy().tobytes()
                                      .decode("utf-8"))
                    if _md.get("groupsize"):
                        g = int(_md["groupsize"])
                except Exception:      # noqa: BLE001
                    pass
            parent, name = cd.rsplit(".", 1)
            mod = model.get_submodule(parent)
            bias = getattr(mod, "bias", None)
            if isinstance(bias, _t.Tensor) and bias.device.type == "meta":
                bias = None
            setattr(mod, name, _Int8DequantLinear(
                w, s, bias=bias, convrot_groupsize=g))
    n_int8 = sum(1 for m in model.modules() if isinstance(m, _Int8DequantLinear))
    print(f"[wan][int8] 已替换 {n_int8} 层为 int8-dequant "
          f"{time.time()-t0:.1f}s", flush=True)

    _ps = list(model.parameters())
    _meta = sum(1 for _prm in _ps if _prm.device.type == "meta")
    n_bytes = sum(b.numel() * b.element_size()
                  for b in model.buffers() if b.device.type != "meta")
    n_bytes += sum(p.numel() * p.element_size()
                   for p in _ps if p.device.type != "meta")
    print(f"[wan][int8] 就绪 meta={_meta} 参数{sum(p.numel() for p in _ps)/1e9:.2f}B "
          f"权重≈{n_bytes/2**30:.1f}GB {time.time()-t0:.1f}s", flush=True)
    if _meta:
        raise RuntimeError(f"wan[int8]: {_meta} 个参数停在 meta")
    return model

# ===================== Wan-Animate 独立装配（2026-11-16） =====================
# **为什么独立装配**：3090 上"模型权重/VAE/TE/CLIP/GPU 全部实测正常"，且
# diffusers 官方最简脚本（自己 from_pretrained 各组件 → 自己 apply_group_
# offloading → 自己 new WanAnimatePipeline → pipe(...)）能正常出片；而经
# `wan-pipe` 装配出的 pipeline 无论怎么调参都输出彩色雪花。差异只在装配
# 路径，因此这里**逐行照抄官方脚本**，绕开 _op_pipe 的组件缓存/附加设置。
_STANDALONE_PIPE = {}


def _assemble_animate_standalone(transformer_dir, te_dir, vae_file,
                                 img_enc_file, flow_shift, te_on_cuda,
                                 skip_te=False, prompt_embeds_path="",
                                 negative_embeds_path="", lora_config=""):
    import torch
    from diffusers import (AutoencoderKLWan, WanAnimatePipeline,
                           WanAnimateTransformer3DModel)
    from diffusers.hooks import apply_group_offloading
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from safetensors.torch import load_file
    from transformers import (AutoTokenizer, CLIPImageProcessor,
                              CLIPVisionConfig, CLIPVisionModel,
                              UMT5EncoderModel)

    key = (transformer_dir, te_dir, vae_file, img_enc_file,
           float(flow_shift), bool(te_on_cuda), bool(skip_te), lora_config)
    if key in _STANDALONE_PIPE:
        print("[wan][standalone] 复用已装配 pipeline", flush=True)
        return _STANDALONE_PIPE[key]

    DT = torch.bfloat16
    t0 = time.time()
    tr = WanAnimateTransformer3DModel.from_pretrained(
        transformer_dir, torch_dtype=DT, low_cpu_mem_usage=True)
    _ps = list(tr.parameters())
    _meta = sum(1 for _p in _ps if _p.device.type == "meta")
    if _meta:
        raise RuntimeError(
            f"wan[standalone]: transformer {_meta}/{len(_ps)} 参数在 meta")
    print(f"[wan][standalone] transformer {time.time()-t0:.1f}s "
          f"meta={_meta}", flush=True)

    if skip_te:
        # 预计算向量命中：完全不加载 11GB UMT5（省显存 + 省 10~20s）
        # ⚠ 不能传 None：pipeline 内部 `dtype = dtype or self.text_encoder.dtype`
        # 会崩（'NoneType' object has no attribute 'dtype'），且 _execution_device
        # 也会遍历组件。这里给一个**零参数占位模块**，只提供 dtype/to/eval 接口。
        tok = AutoTokenizer.from_pretrained(
            os.path.join(os.path.dirname(te_dir.rstrip("/")), "tokenizer"))

        class _TEStub(torch.nn.Module):
            """零显存占位：持有 dtype，前向直接报错（正常路径不该走到）。"""

            def __init__(self):
                super().__init__()
                self.register_buffer("_dtype_anchor",
                                     torch.zeros(1, dtype=torch.bfloat16),
                                     persistent=False)

            @property
            def dtype(self):
                return self._dtype_anchor.dtype

            def forward(self, *a, **k):
                raise RuntimeError(
                    "wan: text_encoder 占位模块被调用——本路径应直接使用"
                    "预计算 prompt_embeds（检查 wan-te-cache 输出是否正确接线）")

        te = _TEStub()
        print("[wan][standalone] 使用预计算文本向量 → 跳过 UMT5 加载"
              "（text_encoder 以零参数占位）", flush=True)
    else:
        tok = AutoTokenizer.from_pretrained(
            os.path.join(os.path.dirname(te_dir.rstrip("/")), "tokenizer"))
        te = UMT5EncoderModel.from_pretrained(te_dir, torch_dtype=DT)

    vae = AutoencoderKLWan.from_single_file(vae_file, torch_dtype=torch.float32)

    # CLIP 手动构造（官方脚本同款：不依赖 config 文件，逐键剥 vision_model.）
    _cfg = CLIPVisionConfig(hidden_size=1280, image_size=224,
                            intermediate_size=5120, num_attention_heads=16,
                            num_hidden_layers=32, patch_size=14,
                            projection_dim=1024)
    clip = CLIPVisionModel(_cfg)
    _csd = load_file(img_enc_file)
    _csd = {(k[len("vision_model."):] if k.startswith("vision_model.") else k): v
            for k, v in _csd.items()}
    _csd = {k: v for k, v in _csd.items()
            if not k.startswith(("visual_projection", "logit_scale",
                                 "text_model")) and k != "position_ids"}
    _m, _u = clip.load_state_dict(_csd, strict=False)
    print(f"[wan][standalone] clip missing={len(_m)} unexpected={len(_u)}",
          flush=True)
    clip = clip.to(torch.float32)

    iproc = CLIPImageProcessor()          # 官方脚本用默认值
    sched = FlowMatchEulerDiscreteScheduler(
        shift=float(flow_shift), num_train_timesteps=1000)

    pipe = WanAnimatePipeline(tokenizer=tok, text_encoder=te, vae=vae,
                              scheduler=sched, image_processor=iproc,
                              image_encoder=clip, transformer=tr)
    del tok

    tr2 = pipe.transformer.to("cpu")
    torch.cuda.empty_cache()
    # **LoRA / 打光合并（在挂 offload 之前，权重仍为 CPU 内存态）**
    if lora_config:
        _apply_lora_config(pipe, lora_config)
    apply_group_offloading(
        tr2, onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"), offload_type="block_level",
        num_blocks_per_group=1, use_stream=False, low_cpu_mem_usage=True)
    pipe.vae.to("cuda")
    # **VAE tiling（2026-11-16 长视频）**：124 帧 512×288 的整段解码会把
    # 24GB 显存打爆（服务被 OOM killer 重启，job 随之丢失 → 节点报 404）。
    # 分块解码把峰值降到常数级，代价是解码稍慢。
    try:
        pipe.vae.enable_tiling()
        print("[wan][standalone] VAE tiling on（长视频分块解码）", flush=True)
    except Exception as _te2:  # noqa: BLE001
        print(f"[wan][standalone] VAE tiling 不可用: {type(_te2).__name__}",
              flush=True)
    if te_on_cuda:
        pipe.text_encoder.to("cuda")
        pipe.image_encoder.to("cuda")
    else:
        # **TE/CLIP 也走块级 offload（2026-11-16 长视频）**：常驻要 12GB，
        # 长视频（124 帧）的 KV cache 需要这部分显存。挂 group_offloading
        # 后前向时逐块上卡，峰值只剩单块 + 激活。
        for _cn, _npg in (("text_encoder", 2), ("image_encoder", 1)):
            _cc = getattr(pipe, _cn, None)
            if _cc is None:
                continue
            apply_group_offloading(
                _cc, onload_device=torch.device("cuda"),
                offload_device=torch.device("cpu"),
                offload_type="block_level", num_blocks_per_group=_npg,
                use_stream=False, low_cpu_mem_usage=True)
        print("[wan][standalone] TE/CLIP 走块级 offload（省 12GB 给 KV cache）",
              flush=True)
    print(f"[wan][standalone] 装配完成 {time.time()-t0:.1f}s "
          f"(TE/CLIP {'cuda' if te_on_cuda else 'offload-cpu'}, "
          f"shift={float(flow_shift)})", flush=True)
    _STANDALONE_PIPE[key] = pipe
    return pipe
