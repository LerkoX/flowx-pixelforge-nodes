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


OP_VERSION = "b257222fdc4f"
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
_VACE_CFG = {
    "14b": {
        "_class_name": "WanVACETransformer3DModel",
        "added_kv_proj_dim": 1536,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "dim": 5120,
        "eps": 1e-06,
        "ffn_dim": 13824,
        "freq_dim": 256,
        "in_channels": 96,
        "num_attention_heads": 40,
        "num_layers": 40,
        "out_channels": 16,
        "patch_size": [1, 2, 2],
        "qk_norm": "rms_norm_across_heads",
        "rope_max_seq_len": 1024,
        "text_dim": 4096,
    },
    "1.3b": {
        "_class_name": "WanVACETransformer3DModel",
        "added_kv_proj_dim": 1536,
        "attention_head_dim": 128,
        "cross_attn_norm": True,
        "dim": 1536,
        "eps": 1e-06,
        "ffn_dim": 8960,
        "freq_dim": 256,
        "in_channels": 96,
        "num_attention_heads": 12,
        "num_layers": 30,
        "out_channels": 16,
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
        path, config=cfg_dir, dtype=torch_dtype)
    n = sum(p.numel() for p in model.parameters())
    print(f"[wan] {which} ready {n/1e9:.2f}B params "
          f"{time.time()-t0:.1f}s", flush=True)
    return model


def _build_animate2_transformer(path, dtype, which="transformer"):
    """Wan-Animate-2 transformer（原始 WanAnimateModel → diffusers）。

    共享盘/官方发布的 `Wan2.2-Animate-14B` 是**原始格式**（config.json 的
    `_class_name` 是 `WanAnimateModel`，含 `motion_encoder_dim`），不是 diffusers
    结构。diffusers 0.42 提供 `convert_wan_animate_2_transformer_to_diffusers`
    做 key 映射，我们复用它。

    Animate-2 与 v1 的差别：**不需要 face_video**（只吃角色图 + 动作视频），
    接口更简单。
    """
    import glob as _glob
    import json as _json
    import time as _time

    import torch
    from diffusers import WanAnimate2Transformer3DModel
    from diffusers.loaders.single_file_utils import (
        convert_wan_animate_2_transformer_to_diffusers)
    from safetensors.torch import load_file

    t0 = _time.time()
    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]

    if os.path.isdir(path):
        cfg_p = os.path.join(path, "config.json")
        shards = sorted(_glob.glob(os.path.join(
            path, "diffusion_pytorch_model*.safetensors")))
        if not shards:
            shards = sorted(_glob.glob(os.path.join(path, "*.safetensors")))
        if not (os.path.isfile(cfg_p) and shards):
            raise ValueError(f"wan: animate2 目录缺 config.json 或权重分片（{path}）")
        orig_cfg = _json.load(open(cfg_p, encoding="utf-8"))
    else:
        raise ValueError(
            f"wan: animate2 需要目录（原始发布格式），got 单文件 {path}")

    print(f"[wan] animate2 读 {len(shards)} 个分片（原始 WanAnimateModel 格式）",
          flush=True)
    sd = {}
    for sh in shards:
        sd.update(load_file(sh))
    print(f"[wan] animate2 原始 keys={len(sd)}", flush=True)

    # 原始 config → diffusers transformer config（字段名不同）
    dim = orig_cfg.get("dim", 5120)
    n_heads = orig_cfg.get("num_heads", 40)
    dcfg = {
        "_class_name": "WanAnimate2Transformer3DModel",
        "added_kv_proj_dim": orig_cfg.get("added_kv_proj_dim"),
        "attention_head_dim": dim // n_heads if n_heads else 128,
        "cross_attn_norm": orig_cfg.get("cross_attn_norm", True),
        "dim": dim,
        "eps": orig_cfg.get("eps", 1e-6),
        "ffn_dim": orig_cfg.get("ffn_dim", 13824),
        "freq_dim": orig_cfg.get("freq_dim", 256),
        "in_channels": orig_cfg.get("in_dim", 36),
        "num_attention_heads": n_heads,
        "num_layers": orig_cfg.get("num_layers", 40),
        "out_channels": 16,
        "patch_size": orig_cfg.get("patch_size", [1, 2, 2]),
        "qk_norm": orig_cfg.get("qk_norm", "rms_norm_across_heads"),
        "rope_max_seq_len": orig_cfg.get("rope_max_seq_len", 1024),
        "text_dim": orig_cfg.get("text_dim", 4096),
    }
    model = WanAnimate2Transformer3DModel.from_config(dcfg)
    try:
        sd = convert_wan_animate_2_transformer_to_diffusers(sd)
        print(f"[wan] animate2 转换后 keys={len(sd)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[wan] animate2 转换函数失败（{type(e).__name__}: {str(e)[:120]}）"
              f"，尝试直接 load", flush=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[wan] ⚠ animate2 missing {len(missing)} keys"
              f"（样例 {missing[:5]}）", flush=True)
    if unexpected:
        print(f"[wan] ⚠ animate2 unexpected {len(unexpected)} keys"
              f"（样例 {unexpected[:5]}）", flush=True)
    model = model.to(torch_dtype)
    print(f"[wan] animate2 transformer ready {_time.time()-t0:.1f}s", flush=True)
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
        # Diffusers 目录（官方 *-Diffusers 仓库）：config 已随目录提供
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
        path, config=cfg_dir, dtype=torch_dtype)
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
            path, config=cfg_dir, dtype=torch.float32)
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
    for prefix in ("vision_model.", "image_encoder.vision_model.",
                   "model.vision_model."):
        if all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}
            break
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:
        print(f"[wan] image_encoder missing {len(missing)} keys", flush=True)
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

    cfg = _config_json("scheduler") or {}
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
    from diffusers import (WanAnimate2ModularPipeline, WanAnimatePipeline,
                           WanImageToVideoPipeline, WanPipeline,
                           WanVACEPipeline)

    common = dict(tokenizer=te_info["tokenizer"], text_encoder=te_info["model"],
                  vae=vae_info["vae"], scheduler=scheduler,
                  transformer=tr_info["model"])
    if tr_info.get("is_moe") and tr_info.get("model2") is not None:
        common["transformer_2"] = tr_info["model2"]
        common["boundary_ratio"] = float(boundary_ratio)

    if task == "animate2":
        # Animate-2 是 modular pipeline（与 H3 同架构），组件用 register 而非构造参数
        pipe = WanAnimate2ModularPipeline()
        pipe.register_components(
            tokenizer=te_info["tokenizer"], text_encoder=te_info["model"],
            vae=vae_info["vae"], scheduler=scheduler,
            transformer=tr_info["model"])
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


def _load_components(transformer, transformer2, text_encoder, vae,
                     image_encoder, dtype, offload, key_prefix="wan:load"):
    """加载全部组件（一体版与拆分版共用），返回 (tr_info, te_info,
    vae_info, img_info, cache_key, weight_gb)。"""
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tr_path = _resolve("diffusion_models", transformer)
    _require_file(tr_path, "transformer")
    low_name = transformer.lower()
    is_animate2 = "animate" in low_name and (
        "animate-2" in low_name or "animate2" in low_name or "animate_2" in low_name)
    is_vace = "vace" in low_name
    if is_animate2:
        # Animate-2：原始 WanAnimateModel 格式 → diffusers（官方转换函数）
        tr = _build_animate2_transformer(tr_path, dtype, "transformer")
        print("[wan] 检测到 Animate-2 权重 → Animate-2 装配路径", flush=True)
    elif is_vace:
        # VACE 用专用 transformer（in_channels=96 + added_kv），不能用普通类
        tr = _build_vace_transformer(tr_path, dtype, "transformer")
        print("[wan] 检测到 VACE transformer → VACE 装配路径", flush=True)
    else:
        tr = _build_transformer(tr_path, dtype, "transformer")
    tr = _offload(tr, device, offload, "transformer")
    tr2 = None
    if transformer2:
        tr2_path = _resolve("diffusion_models", transformer2)
        _require_file(tr2_path, "transformer2")
        tr2 = _build_transformer(tr2_path, dtype, "transformer_2")
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

    @registry.register(
        "wan.load",
        inputs={"task": "STRING", "transformer": "STRING",
                "transformer2": "STRING", "text_encoder": "STRING",
                "vae": "STRING", "image_encoder": "STRING",
                "lora": "STRING", "lora_scale": "FLOAT",
                "dtype": "STRING", "offload": "STRING",
                "boundary_ratio": "FLOAT", "flow_shift": "FLOAT",
                "vae_tiling": "BOOL", "vae_slicing": "BOOL",
                "attention": "STRING", "max_sequence_length": "INT",
                "scheduler": "STRING"},
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
                 flow_shift=5.0, vae_tiling=False, vae_slicing=False,
                 attention="default", max_sequence_length=512,
                 scheduler="euler"):
        task = (task or "t2v").lower()
        if task not in ("t2v", "i2v", "animate", "animate2", "vace"):
            raise ValueError(
                f"wan.load: task 须为 t2v/i2v/animate/animate2/vace，得 {task}")
        is_moe = bool(transformer2)
        key = (f"wan:{task}:{transformer}|{transformer2}|{text_encoder}|{vae}|"
               f"{image_encoder}|{lora}@{lora_scale}|{dtype}|{offload}|"
               f"{boundary_ratio}|{flow_shift}|{lora}@{lora_scale}|"
               f"{vae_tiling}|{vae_slicing}|"
               f"{attention}|{max_sequence_length}")
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                print(f"[wan] cache hit '{key[:40]}…'", flush=True)
                return {"model": models._pipes[key]}

        tr_info, te_info, vae_info, img_info, gb = _load_components(
            transformer, transformer2, text_encoder, vae, image_encoder,
            dtype, offload)
        sched = _build_scheduler(flow_shift, scheduler)
        pipe = _assemble_pipe(
            task, tr_info, te_info, vae_info, img_info, sched, boundary_ratio,
            extra={"vae_tiling": vae_tiling, "vae_slicing": vae_slicing,
                   "attention": attention})

        # LoRA（Wan2.2 MoE 要同时挂两个专家才有效）
        if lora:
            pipe = _load_lora(pipe, lora, lora_scale,
                              into_transformer_2=is_moe)

        pipe._wan_fps = _guess_fps(transformer)
        pipe._wan_max_seq = int(max_sequence_length)
        print(f"[wan] pipe assembled ({task}"
              f"{', MoE' if is_moe else ''}, offload={offload}, "
              f"权重 {gb/1e9:.1f}GB)", flush=True)
        # est=0：组件已各自记账（与 H3 同口径，避免 vram 闸门误判）
        return {"model": _register(
            key, pipe, 0,
            f"Wan{'Animate' if task == 'animate' else 'Pipeline'}")}

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
        tr = _build_transformer(tr_path, dtype, "transformer")
        tr = _offload(tr, device, offload, "transformer")
        tr2 = None
        if transformer2:
            tr2_path = _resolve("diffusion_models", transformer2)
            _require_file(tr2_path, "transformer2")
            tr2 = _build_transformer(tr2_path, dtype, "transformer_2")
            tr2 = _offload(tr2, device, offload, "transformer_2")

        info = {"model": tr, "model2": tr2, "is_moe": bool(transformer2)}
        est = _size(tr_path) + (_size(_resolve("diffusion_models",
                                               transformer2))
                                if transformer2 else 0)
        return {"transformer": _register(key, info, est,
                                         "WanTransformer3DModel",
                                         offload_mode=offload)}

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
                "scheduler": "STRING"},
        outputs={"model": "MODEL"},
        description="Wan 管道组装：把 transformer/text_encoder/vae/"
                    "image_encoder 组件装成可生成管道（task=t2v/i2v/animate）。"
                    "换 VAE 或采样参数不必重装 28GB transformer。")
    def _op_pipe(task="t2v", transformer=None, text_encoder=None, vae=None,
                 image_encoder=None, boundary_ratio=0.875, flow_shift=5.0,
                 scheduler="euler", lora="", lora_scale=1.0,
                 vae_tiling=False, vae_slicing=False, attention="default",
                 max_sequence_length=512):
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
                   "attention": attention})
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
        "wan.gen",
        inputs={"model": "MODEL", "prompt": "STRING",
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
                    "与 H3 不同。帧数自动对齐 4k+1，fps=16（Wan2.1）/24"
                    "（Wan2.2-TI2V-5B）。无音轨。分钟级任务，请经 /jobs。")
    def _op_gen(model, prompt="", negative_prompt="", image=None,
                last_image=None, num_frames=81, height=480, width=832,
                steps=30, guidance_scale=5.0, guidance_scale_2=0.0,
                num_videos_per_prompt=1, max_sequence_length=0, fps=0,
                seed=-1):
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

        fps_out = int(fps) if fps and fps > 0 else int(
            getattr(pipe, "_wan_fps", WAN_FPS_DEFAULT))
        msl = int(max_sequence_length) or int(
            getattr(pipe, "_wan_max_seq", 512))
        kwargs = dict(prompt=prompt, negative_prompt=negative_prompt or None,
                      num_frames=aligned, height=h, width=w,
                      num_inference_steps=int(steps),
                      guidance_scale=float(guidance_scale),
                      num_videos_per_prompt=max(1, int(num_videos_per_prompt)),
                      max_sequence_length=msl,
                      generator=gen, output_type="np")
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
        else:
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
        "wan.animate2",
        inputs={"model": "MODEL", "image": "IMAGE", "pose_video": "VIDEO",
                "prompt": "STRING", "negative_prompt": "STRING",
                "pose_strength": "FLOAT", "pose_start_percent": "FLOAT",
                "pose_end_percent": "FLOAT",
                "reference_image_strength": "FLOAT",
                "segment_frame_length": "INT", "height": "INT",
                "width": "INT", "steps": "INT", "guidance_scale": "FLOAT",
                "seed": "INT"},
        outputs={"video": "VIDEO", "seed": "INT", "fps": "INT"},
        description="Wan-Animate-**2**（角色动画/替换，**不需要 face_video**）："
                    "给角色图 image + 动作视频 pose_video（建议用 "
                    "video-control-preprocess 的 mode=pose 产出**骨架视频**，"
                    "这是 ComfyUI WanAnimate 的核心手法：显式骨骼信号保证姿态"
                    "准确与帧间稳定）。精细旋钮：pose_strength 姿态强度、"
                    "pose_start/end_percent 姿态生效的采样区间（0.7 可放松细节"
                    "保留编排）、reference_image_strength 身份约束强度。")
    def _op_animate2(model, image=None, pose_video=None, prompt="",
                     negative_prompt="", pose_strength=1.0,
                     pose_start_percent=0.0, pose_end_percent=1.0,
                     reference_image_strength=1.0, segment_frame_length=81,
                     height=0, width=0, steps=20, guidance_scale=1.0,
                     seed=-1):
        import random

        import numpy as np
        import torch
        from PIL import Image as _Image

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
            if isinstance(ref, _Image.Image):
                w, h = ref.size
            else:
                a = np.asarray(ref)
                h, w = a.shape[0], a.shape[1]
        mod = 16
        h = max(mod, round(h / mod) * mod)
        w = max(mod, round(w / mod) * mod)

        fps = int(getattr(pipe, "_wan_fps", WAN_FPS_DEFAULT))
        kwargs = dict(image=ref, pose_video=pose, prompt=prompt,
                      negative_prompt=negative_prompt or None,
                      height=h, width=w,
                      num_inference_steps=int(steps),
                      guidance_scale=float(guidance_scale),
                      generator=gen, output_type="np")
        # Animate-2 的精细旋钮（diffusers 有则传，无则忽略）
        for k, v in (("pose_strength", pose_strength),
                     ("pose_start_percent", pose_start_percent),
                     ("pose_end_percent", pose_end_percent),
                     ("reference_image_strength", reference_image_strength)):
            if float(v) != 1.0 and k != "pose_start_percent" or \
                    k == "pose_start_percent" and float(v) != 0.0:
                kwargs[k] = float(v)
        t0 = time.time()
        out = pipe(**kwargs)
        arr = np.asarray(out.frames[0])
        print(f"[wan] animate2 frames shape={arr.shape} "
              f"min={float(arr.min()):.3f} max={float(arr.max()):.3f}",
              flush=True)
        if arr.ndim == 4 and arr.shape[1] in (1, 3) and arr.shape[-1] not in (1, 3):
            arr = np.transpose(arr, (0, 2, 3, 1))
        if arr.min() < -0.05:
            arr = (arr + 1.0) / 2.0
        frames = [_Image.fromarray((np.clip(f, 0, 1) * 255).round().astype("uint8"))
                  for f in arr]
        print(f"[wan] animate2 done {len(frames)}帧 {fps}fps "
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
            first = vid[0] if hasattr(vid, "__getitem__") else None
            if first is not None:
                if isinstance(first, Image.Image):
                    w, h = first.size
                else:
                    a = np.asarray(first)
                    h, w = (a.shape[0], a.shape[1]) if a.ndim >= 2 else (480, 832)
            else:
                w, h = 832, 480
        mod = 16
        h = max(mod, round(h / mod) * mod)
        w = max(mod, round(w / mod) * mod)

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
