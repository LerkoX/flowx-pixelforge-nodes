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

算子：
- wan.load(task, transformer, transformer2, text_encoder, vae, image_encoder,
  lora, lora_scale, dtype, offload, boundary_ratio, config_dir)
  → MODEL（task = t2v | i2v | animate）
- wan.gen(model, prompt, negative_prompt, image?, last_image?, num_frames,
  height, width, steps, guidance_scale, seed, flow_shift)
  → VIDEO（frames+fps，无音轨）+ seed；分钟级任务，请经 POST /jobs 异步执行
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


OP_VERSION = "33dc8a13b130"
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


def _register(key, obj, est_bytes, arch):
    models = _models()
    with models._lock:
        models._evict_if_needed(est_bytes)
        models._pipes[key] = obj
        models._archs[key] = arch
        models._sizes[key] = est_bytes
        models._last_used[key] = time.time()
    print(f"[wan] registered '{key}' est={est_bytes/1e9:.1f}GB", flush=True)
    return obj


def _require_file(path, what):
    if not os.path.isfile(path):
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
    """WAN_DIR 下解析权重路径：支持子目录、绝对路径、裸文件名。"""
    if not name:
        return ""
    if os.path.isabs(name):
        return name
    p = os.path.join(WAN_DIR, subdir, name)
    if os.path.exists(p):
        return p
    p2 = os.path.join(WAN_DIR, name)
    return p2 if os.path.exists(p2) else p


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
    """读 WAN_CONFIG_DIR/<name>/config.json；不存在返回 None。"""
    p = os.path.join(WAN_CONFIG_DIR, name, "config.json")
    if os.path.isfile(p):
        import json
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
    if offload == "sequential":
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
    cfg = _config_json(which)
    kwargs = {"dtype": torch_dtype}
    if cfg is not None:
        print(f"[wan] {which} config 来自 {WAN_CONFIG_DIR}/{which}", flush=True)
        kwargs["config"] = cfg
    try:
        model = WanTransformer3DModel.from_single_file(path, **kwargs)
    except Exception as e:  # noqa: BLE001
        if cfg is not None:
            raise
        arch = _guess_arch(os.path.basename(path))
        print(f"[wan] {which} 无 config，按 {arch} 结构兜底"
              f"（{type(e).__name__}: {str(e)[:120]}）", flush=True)
        model = WanTransformer3DModel.from_single_file(
            path, config=dict(_TRANSFORMER_CFG[arch]), **kwargs)
    n = sum(p.numel() for p in model.parameters())
    print(f"[wan] {which} ready {n/1e9:.2f}B params "
          f"{time.time()-t0:.1f}s", flush=True)
    return model


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
    cfg = _config_json("vae") or dict(_VAE_CFG)
    try:
        vae = AutoencoderKLWan.from_single_file(
            path, config=cfg, dtype=torch.float32)
    except Exception:  # noqa: BLE001
        from diffusers import AutoModel
        vae = AutoModel.from_single_file(path, dtype=torch.float32)
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
    for prefix in ("vision_model.", "image_encoder.vision_model.",
                   "model.vision_model."):
        if all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
            break
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[wan] image_encoder missing {len(missing)} keys", flush=True)
    return m.to(torch.float32)


def _build_scheduler(flow_shift):
    """UniPCMultistepScheduler + flow_shift（720p→5.0，480p→3.0）。"""
    from diffusers.schedulers.scheduling_unipc_multistep import (
        UniPCMultistepScheduler)

    cfg = _config_json("scheduler") or {}
    cfg.update({"num_train_timesteps": 1000, "flow_shift": float(flow_shift)})
    return UniPCMultistepScheduler.from_config(cfg)


def _align_frames(n):
    """Wan 帧数契约 **4k+1**（81/121/161…）。"""
    n = max(5, int(n))
    k = max(1, round((n - 1) / 4))
    return 4 * k + 1


# ---------------------------------------------------------------------------
# 算子注册
# ---------------------------------------------------------------------------

def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "wan.load",
        inputs={"task": "STRING", "transformer": "STRING",
                "transformer2": "STRING", "text_encoder": "STRING",
                "vae": "STRING", "image_encoder": "STRING",
                "lora": "STRING", "lora_scale": "FLOAT",
                "dtype": "STRING", "offload": "STRING",
                "boundary_ratio": "FLOAT", "flow_shift": "FLOAT"},
        outputs={"model": "MODEL"},
        description="Wan 装配：task=t2v（WanPipeline）/ i2v（WanImageToVideo"
                    "Pipeline，给 last_image 即首尾帧）/ animate（WanAnimate"
                    "Pipeline 角色动画+替换）。transformer2 非空即 Wan2.2 MoE"
                    " 双专家（boundary_ratio 处从高噪切低噪）。14B bf16 "
                    "~28GB 走逐层 offload 在 24GB 卡上跑。")
    def _op_load(task="t2v",
                 transformer="wan2.1_t2v_14B_bf16.safetensors",
                 transformer2="",
                 text_encoder="umt5_xxl_bf16.safetensors",
                 vae="wan_2.1_vae.safetensors",
                 image_encoder="",
                 lora="", lora_scale=1.0, dtype="bf16",
                 offload="sequential", boundary_ratio=0.875,
                 flow_shift=5.0):
        import torch

        task = (task or "t2v").lower()
        if task not in ("t2v", "i2v", "animate"):
            raise ValueError(f"wan.load: task 须为 t2v/i2v/animate，得 {task}")
        is_moe = bool(transformer2)
        key = (f"wan:{task}:{transformer}|{transformer2}|{text_encoder}|{vae}|"
               f"{image_encoder}|{lora}@{lora_scale}|{dtype}|{offload}|"
               f"{boundary_ratio}|{flow_shift}")
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                print(f"[wan] cache hit '{key[:40]}…'", flush=True)
                return {"model": models._pipes[key]}

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tr_path = _resolve("diffusion_models", transformer)
        _require_file(tr_path, "transformer")

        # ① transformer（+ MoE 低噪专家）
        tr = _build_transformer(tr_path, dtype, "transformer")
        tr = _offload(tr, device, offload, "transformer")
        tr2 = None
        if is_moe:
            tr2_path = _resolve("diffusion_models", transformer2)
            _require_file(tr2_path, "transformer2")
            tr2 = _build_transformer(tr2_path, dtype, "transformer_2")
            tr2 = _offload(tr2, device, offload, "transformer_2")

        # ② text encoder
        te_path = _resolve("text_encoders", text_encoder)
        _require_file(te_path, "text_encoder")
        te = _build_text_encoder(te_path, dtype)
        te = _offload(te, device, offload, "text_encoder")

        # ③ vae（fp32）
        vae_path = _resolve("vae", vae)
        _require_file(vae_path, "vae")
        vae_m = _build_vae(vae_path)
        if device.type == "cuda":
            vae_m = vae_m.to(device)

        # ④ tokenizer（umt5 词表；共享盘/本地 config 目录均可）
        tok_src = os.path.join(WAN_CONFIG_DIR, "tokenizer")
        if not os.path.isdir(tok_src):
            tok_src = os.path.join(WAN_CONFIG_DIR, "tokenizer_2")
        if not os.path.isdir(tok_src):
            raise ValueError(f"wan: 缺 tokenizer 目录（{WAN_CONFIG_DIR}/"
                             f"tokenizer）——umt5 词表须从官方 Diffusers "
                             f"仓库 copy，KB 级")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(tok_src)

        sched = _build_scheduler(flow_shift)

        # ⑤ 管道装配
        from diffusers import (WanAnimatePipeline, WanImageToVideoPipeline,
                               WanPipeline)
        common = dict(tokenizer=tokenizer, text_encoder=te, vae=vae_m,
                      scheduler=sched, transformer=tr)
        if tr2 is not None:
            common["transformer_2"] = tr2
            common["boundary_ratio"] = float(boundary_ratio)

        if task == "animate":
            pipe = WanAnimatePipeline(**common)
        elif task == "i2v":
            img_enc = None
            if image_encoder:
                img_enc = _build_image_encoder(
                    _resolve("image_encoders", image_encoder))
                if device.type == "cuda":
                    img_enc = img_enc.to(device)
            pipe = WanImageToVideoPipeline(image_encoder=img_enc, **common)
        else:
            pipe = WanPipeline(**common)

        # ⑥ LoRA（Wan2.2 MoE 要同时挂两个专家才有效）
        if lora:
            pipe = _load_lora(pipe, lora, lora_scale,
                              into_transformer_2=is_moe)

        est = _size(tr_path) + ((_size(_resolve("diffusion_models",
                                               transformer2)))
                                if is_moe else 0)
        print(f"[wan] pipe assembled ({task}"
              f"{', MoE' if is_moe else ''}, offload={offload}, "
              f"权重 {est/1e9:.1f}GB)", flush=True)
        # est=0：组件已各自记账（与 H3 同口径，避免 vram 闸门误判）
        return {"model": _register(key, pipe, 0,
                                   f"Wan{'Animate' if task == 'animate' else 'Pipeline'}")}

    @registry.register(
        "wan.gen",
        inputs={"model": "MODEL", "prompt": "STRING",
                "negative_prompt": "STRING", "image": "IMAGE",
                "last_image": "IMAGE", "num_frames": "INT",
                "height": "INT", "width": "INT", "steps": "INT",
                "guidance_scale": "FLOAT", "seed": "INT"},
        outputs={"video": "VIDEO", "seed": "INT", "fps": "INT"},
        description="Wan 视频生成：无 image → T2V；给 image → I2V；"
                    "同时给 image+last_image → 首尾帧插值。**Wan 有 CFG**"
                    "（guidance_scale>1 时生效，negative_prompt 才有用），"
                    "与 H3 不同。帧数自动对齐 4k+1，fps=16（Wan2.1）/24"
                    "（Wan2.2-TI2V-5B）。无音轨。分钟级任务，请经 /jobs。")
    def _op_gen(model, prompt="", negative_prompt="", image=None,
                last_image=None, num_frames=81, height=480, width=832,
                steps=30, guidance_scale=5.0, seed=-1):
        import random

        import numpy as np
        import torch
        from PIL import Image

        pipe = model
        if not prompt:
            raise ValueError("wan.gen: prompt 必填（Wan 是文本引导模型）")
        if not hasattr(pipe, "transformer"):
            raise ValueError("wan.gen: model 不是 wan.load 的输出")

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

        fps = int(getattr(pipe, "_wan_fps", WAN_FPS_DEFAULT))
        kwargs = dict(prompt=prompt, negative_prompt=negative_prompt or None,
                      num_frames=aligned, height=h, width=w,
                      num_inference_steps=int(steps),
                      guidance_scale=float(guidance_scale), generator=gen,
                      output_type="np")
        t0 = time.time()
        if img is not None:
            # I2V / 首尾帧：WanImageToVideoPipeline 认 image + last_image
            if last is not None:
                kwargs["last_image"] = last
            kwargs["image"] = img
            out = pipe(**kwargs)
        else:
            out = pipe(**kwargs)
        frames_np = out.frames[0]
        frames = [Image.fromarray(
            (np.clip(f, 0, 1) * 255).round().astype("uint8"))
            for f in frames_np]
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
                    segment_frame_length=77, height=0, width=0, steps=20,
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
                return [np.asarray(f.convert("RGB")) for f in pils]
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
        if mode == "replace":
            kwargs["background_video"] = bg
            kwargs["mask_video"] = msk
        t0 = time.time()
        out = pipe(**kwargs)
        frames_np = out.frames[0]
        frames = [_Image.fromarray(
            (np.clip(f, 0, 1) * 255).round().astype("uint8")) for f in frames_np]
        print(f"[wan] animate({mode}) done {len(frames)}帧 {fps}fps "
              f"{time.time()-t0:.1f}s", flush=True)
        return {"video": {"frames": frames, "fps": fps,
                          "num_frames": len(frames)},
                "seed": seed, "fps": fps}
