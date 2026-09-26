"""FLUX.1 算子（GGUF 量化路线，8GB Pascal + 12GB WSL 实测可行）。

背景（Phase 0 实证，dev-plan 二十九节）：
- diffusers 0.35.2 官方支持 FluxPipeline + GGUFQuantizationConfig，但本环境
  三处不兼容需补丁（模块导入时幂等安装）：
  1) torch 2.3.1 无 torch.nn.RMSNorm（2.4 引入，FluxTransformer 需要）
  2) diffusers attention_dispatch 恒传 enable_gqa（torch 2.5 参数）→ SDPA shim
     剥离（False 本就是默认语义；True 明确报错）
  3) transformers 4.44 GGUF 仅支持 llama/mistral/qwen2 → T5-XXL 手工装载
     （gguf dequant → 名称映射 → meta 壳 + load_state_dict(assign=True)）
- **T5-XXL 必须 bf16**：fp16 在其 ~7e5 模长的高激活区精度崩坏（分层对拍
  实证：fp16 24 层后 cos 仅 0.38；bf16 全程 ≥0.88）。
- GGUF shape 反序语义：数据行主序按 [out,in] 存但 shape 记 [in,out]，
  正解 reshape(shape[::-1])（reshape(shape).T 是错误等价，踩过）。
- **内存错峰（本插件核心设计）**：transformer Q4 6.8GB + T5 bf16 4.9GB
  无法同时驻留 WSL 12GB 内存（装载期叠加瞬时即 OOM，实测 exit 137）。
  因此不用 accelerate offload 钩子，插件自管设备相位：
    * 常态：transformer/CLIP/VAE 驻 CPU 内存（~7.5GB）
    * flux.encode：T5 装载上卡（~4.9GB VRAM，此时 transformer 在 CPU
      不占显存）→ 双编码 → 默认释放 T5（release_t5=1）；再次编码自动重载
      （bf16 safetensors 磁盘缓存 ~40s，免 7min GGUF dequant）
    * flux.sample：transformer+VAE 上卡（~7GB VRAM）→ pipe 直调（embedding
      直通，官方调度/packing 路径）→ 完事留卡（连续出图免搬移；encode 来时
      再搬回 CPU）
  代价：encode↔sample 交替时每次 encode 前要把 transformer 搬回 CPU
  （~20s PCIe 拷贝），文档化接受。

权重布局（/models/flux，ModelScope AI-ModelScope/FLUX.1-schnell +
city96 GGUF 混合组装）：
  flux1-schnell-Q4_K_S.gguf   transformer（city96，GGUF Q4_K_S）
  t5xxl-Q4_K_S.gguf           T5-XXL 编码器（city96，GGUF Q4_K_S）
  transformer/config.json      transformer 配置（GGUF from_single_file 需要）
  text_encoder/               CLIP-L（safetensors）
  text_encoder_2/config.json  T5 配置（权重走 GGUF）
  vae/ tokenizer/ tokenizer_2/ scheduler/   其余 diffusers 组件

算子：
- flux.load(transformer, t5, offload) → MODEL（FluxPipeline；offload 旋钮
  保留接口一致，当前实现固定"相位自管"语义，仅校验取值合法）
- flux.encode(model, text, max_seq, release_t5) → COND{prompt_embeds, pooled}
- flux.sample(model, cond, width, height, seed, steps, guidance) → IMAGE
  （schnell 无 CFG 无负向；guidance 仅 dev 版 guidance_embeds=true 时有意义）
"""
import json
import os
import time

import torch

from app import execution, ops, preview

FLUX_DIR = os.environ.get("FLUX_DIR", "/models/flux")
FLUX_CLASS = "FluxPipeline"

# ---------------------------------------------------------------------------
# torch 2.3.1 兼容补丁（幂等，模块导入即安装）
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
# T5-XXL GGUF 手工装载（transformers 4.44 不支持 T5 架构 GGUF）
# llama.cpp/city96 t5encoder 命名 → HF T5 映射；逐张量与官方 safetensors
# 对拍 cos≈1.0 验证（Phase 0 记录）
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
    """llama.cpp/city96 t5encoder gguf 名 → HF T5 名（None=不认识）。"""
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
    """GGUF → HF T5 state_dict。gguf shape 是 [in,out]（ne[] 序），数据行主序
    按 [out,in] 存 → reshape(shape[::-1])（不是 reshape(shape).T！）。
    cache_dir 给定时逐张量 npy 落盘（HF 名）：safetensors save_file 会复制
    出第二份全量副本 → 12GB 内存盒 OOM（实测 exit 137），npy 分片写无全量
    拷贝。device="cuda" 逐张量直上卡：RAM 只吃单张量瞬时。"""
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
    """GGUF T5 → T5EncoderModel（bf16 默认）。meta 壳 + assign=True 避免
    fp32 空壳 OOM。逐张量 npy 缓存（首次 GGUF dequant ~4min，缓存后 ~40s）。
    eval 模式（T5 config dropout=0.1，训练态会让同一 prompt 两次编码结果
    无关——踩过）。
    device="cuda" 时逐张量直上卡（T5 bf16 4.9GB 单独放得下 8GB 显存）：
    transformer 驻留内存（6.8GB）+ T5 驻留显存 → 两相不叠加，12GB WSL
    内存装不下"两者同驻 RAM"（实测 OOM exit 137 两次）。"""
    from transformers import T5Config, T5EncoderModel

    cache_dir = gguf_path + f".{str(dtype).split('.')[-1]}.cache"
    cfg = T5Config(**json.load(open(config_path)))
    if os.path.isfile(os.path.join(cache_dir, _T5_CACHE_MARK)):
        t0 = time.time()
        sd = _t5_state_dict_from_cache(cache_dir, dtype, device)
        print(f"[flux] t5 缓存命中（{time.time()-t0:.1f}s）", flush=True)
    else:
        t0 = time.time()
        # device="cuda" 时不写缓存（cuda 张量落 npy 要先拉回 CPU 形成全量
        # 副本 → OOM）；缓存在空载引导期以 device="cpu" 跑一次生成
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
# 管道装配（相位自管：全部组件驻 CPU 内存，各算子按需上卡）
# ---------------------------------------------------------------------------


def _require_file(path, what):
    if not os.path.isfile(path):
        raise ValueError(
            f"flux: 缺 {what}（{path}）。下载布局见插件 docstring")
    return path


def _load_flux_pipe(transformer="flux1-schnell-Q4_K_S.gguf",
                    t5="t5xxl-Q4_K_S.gguf"):
    from diffusers import (AutoencoderKL, FlowMatchEulerDiscreteScheduler,
                           FluxPipeline, FluxTransformer2DModel,
                           GGUFQuantizationConfig)
    from transformers import (CLIPTextModel, CLIPTokenizer, T5TokenizerFast)

    tr_path = _require_file(os.path.join(FLUX_DIR, transformer),
                            "transformer GGUF")
    _require_file(os.path.join(FLUX_DIR, t5), "T5 GGUF")
    tr_cfg = _require_file(os.path.join(FLUX_DIR, "transformer", "config.json"),
                           "transformer config")

    t0 = time.time()
    quant = GGUFQuantizationConfig(compute_dtype=torch.float16)
    tr = FluxTransformer2DModel.from_single_file(
        tr_path, config=os.path.dirname(tr_cfg),
        quantization_config=quant, torch_dtype=torch.float16)
    print(f"[flux] transformer {os.path.basename(tr_path)} "
          f"{time.time()-t0:.1f}s", flush=True)

    clip = CLIPTextModel.from_pretrained(os.path.join(FLUX_DIR, "text_encoder"),
                                         torch_dtype=torch.bfloat16)
    vae = AutoencoderKL.from_pretrained(os.path.join(FLUX_DIR, "vae"),
                                        torch_dtype=torch.float16)
    pipe = FluxPipeline(
        transformer=tr, vae=vae,
        text_encoder=clip,
        tokenizer=CLIPTokenizer.from_pretrained(
            os.path.join(FLUX_DIR, "tokenizer")),
        text_encoder_2=None,   # T5 编码相位才装载（内存错峰，见 docstring）
        tokenizer_2=T5TokenizerFast.from_pretrained(
            os.path.join(FLUX_DIR, "tokenizer_2")),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            os.path.join(FLUX_DIR, "scheduler")),
    )
    pipe.set_progress_bar_config(disable=True)
    pipe._flux_cfg = {"transformer": transformer, "t5": t5}
    return pipe


def _register_pipe(pipe, key, est_bytes, arch):
    """手工注册进 model_manager LRU（统一显存治理/Unload 联动）。
    单例在 app.main（plugins 扫描发生在其定义之后，运行时取用无循环问题）。"""
    from app.main import models
    with models._lock:
        models._evict_if_needed(est_bytes)
        models._pipes[key] = pipe
        models._archs[key] = arch
        models._sizes[key] = est_bytes
        models._last_used[key] = time.time()
    print(f"[flux] registered '{key}' est={est_bytes/1e9:.1f}GB", flush=True)
    return pipe


def _release_t5(pipe):
    """编码完释放 T5（4.9GB bf16），下次 encode 自动重载（缓存 ~40s）。"""
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2 = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[flux] t5 released (release_t5=1)", flush=True)


def _t5_name_of(pipe):
    return (getattr(pipe, "_flux_cfg", None) or {}).get(
        "t5", "t5xxl-Q4_K_S.gguf")


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------


def register(registry):
    @registry.register(
        "flux.load",
        inputs={"transformer": "STRING", "t5": "STRING", "offload": "STRING"},
        outputs={"model": "MODEL"},
        description="FLUX Loader：GGUF 量化装配 FluxPipeline（transformer Q4_K_S "
                    "fp16-compute + CLIP-L bf16 + VAE fp16；T5 编码相位才装载），"
                    "进 model_manager LRU。offload 旋钮仅校验取值（none/model/"
                    "sequential/auto），当前实现固定相位自管（8GB 卡 GGUF 路线"
                    "的内存错峰见插件 docstring）")
    def _op_flux_load(transformer="flux1-schnell-Q4_K_S.gguf",
                      t5="t5xxl-Q4_K_S.gguf", offload="auto"):
        from app import model_manager as _mm
        from app.main import models
        _mm._resolve_offload(offload)  # 仅校验取值合法
        key = f"flux:{transformer}|t5={t5}"
        if key in models._pipes:
            return {"model": models.get(key)}
        est = os.path.getsize(os.path.join(FLUX_DIR, transformer)) \
            + int(1.5e9)
        pipe = _load_flux_pipe(transformer, t5)
        return {"model": _register_pipe(pipe, key, est, "FluxPipeline(GGUF)")}

    @registry.register(
        "flux.encode",
        inputs={"model": "MODEL", "text": "STRING", "max_seq": "INT",
                "release_t5": "INT"},
        outputs={"cond": "COND"},
        description="FLUX 文本编码：CLIP-L pooled(768) + T5-XXL 序列(256×4096) "
                    "双编码 → COND。T5 走 bf16（fp16 高激活崩坏，勿改）。"
                    "编码相位：transformer 搬回 CPU → T5 上卡 → 编码 → "
                    "release_t5=1（默认）释放 T5")
    def _op_flux_encode(model, text, max_seq=256, release_t5=1):
        pipe, patches = ops.resolve_pipe(model)
        if type(pipe).__name__ != FLUX_CLASS:
            raise ValueError(
                f"flux.encode 只接受 {FLUX_CLASS}（当前 {type(pipe).__name__}）")
        if patches:
            raise ValueError("FLUX LoRA 不支持（MVP 边界）")
        if not torch.cuda.is_available():
            raise ValueError("flux.encode 需要 CUDA（T5 bf16 编码上卡）")

        t0 = time.time()
        # 相位切换（VRAM 侧）：transformer 此刻在 CPU 内存不占显存，
        # T5 逐张量直装上卡（RAM 不吃 4.9GB 瞬时——两者同驻 RAM 会 OOM，
        # 实测 exit 137）；CLIP 上卡
        # transformer/VAE 回 CPU 内存（释放 VRAM 给 T5），再 T5/CLIP 上卡。
        # 用 pipe.to("cpu") 同步 pipe.device 语义（见 flux.sample 注释）
        if pipe.transformer is not None:
            pipe.to("cpu")
        torch.cuda.empty_cache()
        pipe.text_encoder_2 = load_t5_gguf(
            os.path.join(FLUX_DIR, _t5_name_of(pipe)),
            os.path.join(FLUX_DIR, "text_encoder_2", "config.json"),
            device="cuda")
        pipe.text_encoder.to("cuda")
        with torch.no_grad():
            pe, pooled, _ = pipe.encode_prompt(
                prompt=str(text), prompt_2=None, device=torch.device("cuda"),
                max_sequence_length=int(max_seq))
        pipe.text_encoder.to("cpu")
        if int(release_t5):
            _release_t5(pipe)
        else:
            pipe.text_encoder_2.to("cpu")
        print(f"[flux.encode] {time.time()-t0:.1f}s pe={tuple(pe.shape)}",
              flush=True)
        return {"cond": {"prompt_embeds": pe.float().cpu(),
                         "pooled": pooled.float().cpu()}}

    @registry.register(
        "flux.sample",
        inputs={"model": "MODEL", "cond": "COND", "width": "INT", "height": "INT",
                "seed": "INT", "steps": "INT", "guidance": "FLOAT",
                "preview_every": "INT"},
        outputs={"image": "IMAGE", "seed": "INT"},
        description="FLUX Sampler（schnell 4 步蒸馏，无 CFG/负向）：768px/4 步 "
                    "实测 ~29min，512px ~4min（8GB Pascal）。transformer+VAE "
                    "上卡采样，完事留卡（连续出图免搬移）。进度卡片帧留在 "
                    "GET /preview/{job_id}；/interrupt 可取消")
    def _op_flux_sample(model, cond, width=768, height=768, seed=-1, steps=4,
                        guidance=0.0, preview_every=1):
        import random
        pipe, patches = ops.resolve_pipe(model)
        if type(pipe).__name__ != FLUX_CLASS:
            raise ValueError(
                f"flux.sample 只接受 {FLUX_CLASS}（当前 {type(pipe).__name__}）")
        if patches:
            raise ValueError("FLUX LoRA 不支持（MVP 边界）")
        if not isinstance(cond, dict) or "prompt_embeds" not in cond:
            raise ValueError("flux.sample: cond 需为 flux.encode 的输出 "
                             "{prompt_embeds, pooled}")
        if not torch.cuda.is_available():
            raise ValueError("flux.sample 需要 CUDA")

        seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**32 - 1)
        steps, width, height = int(steps), int(width), int(height)
        job = execution.current()
        rec = preview.recorder_for(job.id, int(preview_every)) if job else None

        def _cb(p, i, t, kw):
            execution.check_cancelled()
            if rec:
                rec.push_pil(preview.progress_card(i + 1, steps),
                             (i + 1) / steps)
            return kw

        # 相位切换：整管上卡（transformer 6.8GB + VAE + CLIP ≈ 7.2GB < 8GB）。
        # 不能逐模块 .to：pipe.__call__ 内部用 pipe.device（注册序首个模块）
        # 派生 latent/timestep 设备，逐模块搬移会导致 latent 建在 CPU 而权重
        # 在 GPU（matmul 设备错，实测踩中）；pipe.to() 同步刷新 pipe.device。
        pipe.to("cuda")
        torch.cuda.empty_cache()

        t0 = time.time()
        try:
            img = pipe(
                prompt_embeds=cond["prompt_embeds"].to("cuda", torch.float16),
                pooled_prompt_embeds=cond["pooled"].to("cuda", torch.float16),
                num_inference_steps=steps, guidance_scale=float(guidance),
                width=width, height=height,
                generator=torch.Generator("cpu").manual_seed(seed),
                callback_on_step_end=_cb,
                callback_on_step_end_tensor_inputs=[],
            ).images[0]
        except Exception:
            import traceback
            print("[flux.sample] ERROR:", traceback.format_exc(), flush=True)
            raise
        print(f"[flux.sample] done in {time.time()-t0:.1f}s "
              f"{width}x{height} steps={steps} seed={seed}", flush=True)
        return {"image": img, "seed": seed}
