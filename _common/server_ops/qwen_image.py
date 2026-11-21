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

# --- sync-common 内联点：comfy_quant 通用件（分发时拼接，勿手动复制）---
from comfy_quant import (_sequential_offload)
# --- 内联点结束 ---

OP_VERSION = "07af57eb5a91"
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
    """Qwen2.5-VL-7B 文本编码器（**bf16**，fp16 会溢出）+ tokenizer
    + processor（Edit-2511 的 Qwen2VLProcessor，T2I 仓库没有 → None）。"""
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
    processor = None
    proc_dir = os.path.join(root, "processor")
    if os.path.isdir(proc_dir):
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(proc_dir)
        print("[qwen-image] processor 已加载（EditPlus 需要）", flush=True)
    return te, tokenizer, processor, te_dir


def _build_vae(model, vae, device, dtype="fp32"):
    """AutoencoderKLQwenImage。

    dtype 口径（真机教训）：T2I 官方示例 VAE fp32（解码质量）；但
    **EditPlus 管线的 prepare_latents 会把预处理图像 cast 成 transformer
    dtype（bf16）再 vae.encode** —— fp32 VAE 收到 bf16 输入直接
    'Input type (c10::BFloat16) and bias type (float)'。
    官方 Edit 示例整管 bf16（VAE 同）。故编辑模式 VAE 必须 bf16。"""
    import torch
    from diffusers import AutoencoderKLQwenImage

    dt = torch.float32 if str(dtype).lower() in ("fp32", "float32", "auto") \
        else _dtype(dtype)
    if vae:
        path = _resolve("vae", vae)
        _require_file(path, "vae")
        if os.path.isdir(path):
            m = AutoencoderKLQwenImage.from_pretrained(path, torch_dtype=dt)
        else:
            cfg_dir = os.path.join(_model_root(model), "vae")
            m = AutoencoderKLQwenImage.from_single_file(
                path, config=cfg_dir if os.path.isdir(cfg_dir) else None)
        src = path
    else:
        vae_dir = os.path.join(_model_root(model), "vae")
        _require_file(vae_dir, "vae 目录")
        m = AutoencoderKLQwenImage.from_pretrained(vae_dir, torch_dtype=dt)
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
    # **LoRA 参数必须钉在 GPU（真机教训）**：组件在加载时已挂
    # _sequential_offload hook，PEFT 后注入的 lora_A/lora_B 子模块没有 hook
    # → 留在 CPU；forward 时基础权重被 hook 搬上卡、LoRA 在 CPU →
    # "mat2 is on cpu" 设备冲突。LoRA 参数量小（Lightning ~1GB），
    # 全部常驻 GPU，不参与逐层 swap。
    import torch
    if torch.cuda.is_available():
        n_lora, bytes_lora = 0, 0
        for pname, p in pipe.transformer.named_parameters():
            if "lora" in pname.lower() and p.device.type != "cuda":
                p.data = p.data.to("cuda")
                n_lora += 1
                bytes_lora += p.numel() * p.element_size()
        for bname, b in pipe.transformer.named_buffers():
            if "lora" in bname.lower() and b.device.type != "cuda":
                parent_path, _, attr = bname.rpartition(".")
                parent = (pipe.transformer.get_submodule(parent_path)
                          if parent_path else pipe.transformer)
                parent._buffers[attr] = b.to("cuda")
        print(f"[qwen-image] LoRA 参数钉卡：{n_lora} 个，"
              f"{bytes_lora/2**30:.2f} GiB", flush=True)
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
        te, tokenizer, processor, src = _build_text_encoder(model, dtype)
        te = _offload(te, device, offload, "text_encoder")
        _budget_mode = "sequential" if str(offload).lower() in ("none", "") \
            else offload
        return {"text_encoder": _register(
            key, {"model": te, "tokenizer": tokenizer,
                  "processor": processor}, _size(src),
            "Qwen2_5_VLForConditionalGeneration", offload_mode=_budget_mode)}

    @registry.register(
        "qwen_image.vae_load",
        inputs={"model": "STRING", "vae": "STRING", "device": "STRING",
                "dtype": "STRING"},
        outputs={"vae": "VAE"},
        description="Qwen-Image VAE 加载（AutoencoderKLQwenImage ~243MB，常驻"
                    "上卡）。dtype：T2I 用 fp32（官方口径）；**编辑模式必须 "
                    "bf16**——EditPlus 管线会把输入图 cast 成 transformer "
                    "dtype 再 VAE encode，fp32 VAE 会 dtype 冲突。")
    def _op_vae_load(model="Qwen-Image-2512", vae="", device="cuda",
                     dtype="fp32"):
        key = f"qwen-image:vae:{model}|{vae}|{dtype}"
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"vae": models._pipes[key]}

        m, src = _build_vae(model, vae, device, dtype)
        return {"vae": _register(key, {"vae": m}, _size(src),
                                 "AutoencoderKLQwenImage")}

    @registry.register(
        "qwen_image.pipe",
        inputs={"transformer": "TRANSFORMER", "text_encoder": "TEXT_ENCODER",
                "vae": "VAE", "model": "STRING", "task": "STRING",
                "lora": "STRING",
                "lora_scale": "FLOAT", "max_sequence_length": "INT"},
        outputs={"model": "MODEL"},
        description="Qwen-Image 管道组装：组件 → 管道（+ 可选 Lightning 等 "
                    "LoRA，name@strength 逗号多挂）。task=t2i 文生图 "
                    "QwenImagePipeline；task=edit 图像编辑——有 processor "
                    "用 QwenImageEditPlusPipeline（Edit-2511），否则回退 "
                    "QwenImageEditPipeline（2509）。组件常驻，换 LoRA/参数 "
                    "不必重装 39GB transformer。")
    def _op_pipe(transformer=None, text_encoder=None, vae=None,
                 model="Qwen-Image-2512", task="t2i", lora="",
                 lora_scale=1.0, max_sequence_length=512):
        if transformer is None or text_encoder is None or vae is None:
            raise ValueError("qwen_image.pipe: transformer / text_encoder / "
                             "vae 三个组件必填")
        task = (task or "t2i").lower()
        if task not in ("t2i", "edit"):
            raise ValueError(
                f"qwen_image.pipe: task 须为 t2i/edit，得 {task}")
        tr_info, te_info, vae_info = transformer, text_encoder, vae
        key = (f"qwen-image:pipe:{id(tr_info.get('model')):x}|"
               f"{id(te_info.get('model')):x}|{id(vae_info.get('vae')):x}|"
               f"{model}|{task}|{lora}|{lora_scale}|{max_sequence_length}")
        models = _models()
        with models._lock:
            if key in models._pipes:
                models._last_used[key] = time.time()
                return {"model": models._pipes[key]}

        scheduler = _build_scheduler(model)
        if task == "edit":
            # EditPlus 的 prepare_latents 把输入图 cast 成 transformer dtype
            # 再 vae.encode —— VAE 必须与 transformer 同 dtype（bf16），
            # 否则 fp32 VAE 收 bf16 输入报 bias 类型冲突（真机实证）。
            import torch as _torch
            _vae = vae_info["vae"]
            _vdt = next(_vae.parameters()).dtype
            if _vdt != _torch.bfloat16:
                print(f"[qwen-image] edit 模式 VAE {_vdt} → bf16（跟随 "
                      f"transformer，管线 prepare_latents 口径）", flush=True)
                _vae.to(_torch.bfloat16)
            processor = te_info.get("processor")
            if processor is not None:
                from diffusers import QwenImageEditPlusPipeline
                pipe = QwenImageEditPlusPipeline(
                    scheduler=scheduler, vae=vae_info["vae"],
                    text_encoder=te_info["model"],
                    tokenizer=te_info["tokenizer"], processor=processor,
                    transformer=tr_info["model"])
                pipe_cls = "QwenImageEditPlusPipeline"
            else:
                # 老版 Qwen-Image-Edit（2509）仓库无 processor
                from diffusers import QwenImageEditPipeline
                pipe = QwenImageEditPipeline(
                    scheduler=scheduler, vae=vae_info["vae"],
                    text_encoder=te_info["model"],
                    tokenizer=te_info["tokenizer"],
                    transformer=tr_info["model"])
                pipe_cls = "QwenImageEditPipeline"
        else:
            from diffusers import QwenImagePipeline
            pipe = QwenImagePipeline(
                scheduler=scheduler, vae=vae_info["vae"],
                text_encoder=te_info["model"],
                tokenizer=te_info["tokenizer"],
                transformer=tr_info["model"])
            pipe_cls = "QwenImagePipeline"
        if lora:
            pipe = _load_loras(pipe, lora, lora_scale)
        pipe._qwen_max_seq = int(max_sequence_length)
        print(f"[qwen-image] pipe assembled（{pipe_cls}，model={model}"
              f"{', lora=' + lora if lora else ''}）", flush=True)
        return {"model": _register(key, pipe, 0, pipe_cls)}

    @registry.register(
        "qwen_image.gen",
        inputs={"model": "MODEL", "prompt": "STRING",
                "negative_prompt": "STRING", "steps": "INT",
                "true_cfg_scale": "FLOAT", "width": "INT", "height": "INT",
                "seed": "INT", "max_sequence_length": "INT",
                "num_images": "INT", "image": "IMAGE"},
        outputs={"image": "IMAGE", "seed": "INT"},
        description="Qwen-Image 生成。不给 image → T2I；给 image → 图像编辑"
                    "（须接 task=edit 装配的 pipe）。true_cfg_scale>1 且给了负"
                    "提示才走 CFG（官方默认 4.0）；挂 Lightning LoRA 时用 "
                    "steps=4/8 + true_cfg_scale=1.0。宽高 0=自动（T2I 默认 "
                    "1328×1328，编辑跟随输入图）；>0 时 16 对齐。")
    def _op_gen(model, prompt="", negative_prompt="", steps=50,
                true_cfg_scale=4.0, width=1328, height=1328, seed=-1,
                max_sequence_length=0, num_images=1, image=None):
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
        if img is not None and "Edit" not in type(pipe).__name__:
            raise ValueError(
                f"qwen_image.gen: 给了 image 但管道是 {type(pipe).__name__}"
                f"（非编辑管线）——pipe 请用 task=edit 装配")

        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)

        msl = int(max_sequence_length) or int(
            getattr(pipe, "_qwen_max_seq", 512))
        kwargs = dict(prompt=prompt,
                      negative_prompt=negative_prompt or None,
                      num_inference_steps=int(steps),
                      true_cfg_scale=float(true_cfg_scale),
                      num_images_per_prompt=max(1, int(num_images)),
                      max_sequence_length=msl,
                      generator=gen, output_type="np")
        if img is not None:
            kwargs["image"] = img
        # VAE 8 × patch 2 → 16 对齐；0=自动（T2I 默认 1328，编辑跟随输入图）
        w_in, h_in = int(width), int(height)
        if w_in > 0 and h_in > 0:
            mod = 16
            h = max(mod, round(h_in / mod) * mod)
            w = max(mod, round(w_in / mod) * mod)
            if (h, w) != (h_in, w_in):
                print(f"[qwen-image] {w_in}x{h_in} -> {w}x{h}（16 对齐）",
                      flush=True)
            kwargs["height"], kwargs["width"] = h, w
        elif img is None:
            kwargs["height"], kwargs["width"] = 1328, 1328
            w, h = 1328, 1328
        else:
            w, h = img.size
            print(f"[qwen-image] 宽高跟随输入图 {w}x{h}", flush=True)
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
