"""DDColor 老照片/黑白图上色算子族（G8 / B4 批次，dev-plan §五十 50.2 S1）。

算子：
- `colorize.load`  ：装载 DDColor 变体 → 模型键 `ddcolor:<variant>`（注册进 model_manager LRU）
- `colorize.apply` ：黑白/褪色图 → 上色图（**保留原图 L 通道**，只预测 ab），可选色度增强

**实现来源（vendored，逐行对照上游）**：https://github.com/piddnad/DDColor （2026-10-12 取 master）
- `ddcolor/model.py`（DDColor / ImageEncoder / DuelDecoder / MultiScaleColorDecoder）
- `basicsr/archs/ddcolor_arch_utils/{unet,convnext,transformer_utils,position_encoding}.py`
- `ddcolor/pipeline.py`（`ColorizationPipeline.process` 的数值流程，见 `_infer`）

上述 5 个模块被**原样内联**进 `_ddcolor_classes()` 这个惰性工厂（模块导入时不 import torch，
所以 Termux 本机也能跑纯逻辑单测：nodes/colorize-load/test_main.py）。与上游的唯一差异：
1. 去掉上游的模块级 import 与 `from basicsr...` 交叉引用（改为同一函数作用域内定义 + 内置 import）；
2. `pipeline.py` 最终 `astype(np.uint8)` 前补 `np.clip(…, 0, 1)`：上游对越界值会**回绕**（255→0），
   等价改写只影响越界像素，正常范围像素逐位不变（S1 探针已与上游参考实现对照）。

依赖：torch + `timm`（`convnext.py` 自带纯 torch 回退，timm 只是更快）+ cv2/numpy
⇒ **零新增依赖、不重建推理镜像**。
精度：默认 fp32（Pascal 纪律：GP104 非张量 fp16 吞吐 = fp32 的 1/64，dev-plan §49.2）。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
OP_VERSION = "b77d44da027e"
OP_FAMILY = "colorize"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

import json
import os
import time

from PIL import Image

COLORIZE_DIR = os.environ.get("COLORIZE_MODELS_DIR", "/models/colorize")

# 变体 → 上游仓库（报错时给下载指引）；模型结构以各变体目录里的 config.json 为准。
VARIANTS = {
    "modelscope": {"repo": "piddnad/ddcolor_modelscope",
                   "desc": "DDColor-L 清洗数据版（912MB，默认；非 ImageNet 图效果最好）"},
    "paper": {"repo": "piddnad/ddcolor_paper",
              "desc": "DDColor-L 论文版（912MB）"},
    "artistic": {"repo": "piddnad/ddcolor_artistic",
                 "desc": "DDColor-L 艺术风格版（912MB）"},
    "paper_tiny": {"repo": "piddnad/ddcolor_paper_tiny",
                   "desc": "DDColor-T 最轻量（220MB，convnext-t 编码器）"},
}

# 变体目录必备文件（缺哪个人话点名）
REQUIRED_FILES = ("config.json", "pytorch_model.bin")

_CLASSES = None    # 惰性构造的 vendored 类（首次调用才定义，避免模块级 import torch）
_DTYPES = {}       # 模型键 -> dtype 字符串（淘汰后按原精度重建）
_LOADINFO = {}     # 模型键 -> 装载诊断（权重键匹配情况等）


# ---------------------------------------------------------------------------
# 目录 / 模型构造 / LRU 记账
# ---------------------------------------------------------------------------

def _variant_dir(base, variant, op_name):
    """变体目录定位 + 必备文件检查（纯 os 操作，先快速失败）。"""
    if variant not in VARIANTS:
        raise ValueError(f"{op_name}: variant 非法 {variant!r}，可选 "
                         f"{sorted(VARIANTS)}")
    path = os.path.join(base, variant)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{op_name}: 变体目录不存在 {path}（下载 {VARIANTS[variant]['repo']} "
            f"到该目录；环境变量指向 {base}）")
    missing = [f for f in REQUIRED_FILES
               if not os.path.isfile(os.path.join(path, f))]
    if missing:
        raise FileNotFoundError(
            f"{op_name}: 变体 {variant} 权重缺失 {path}/{'、'.join(missing)}"
            f"（需 config.json + pytorch_model.bin，来自 "
            f"{VARIANTS[variant]['repo']}；下载见 nodes/README.md 权重清单）")
    return path


def _dtype(name, op_name):
    import torch
    s = str(name or "fp32").strip().lower()
    if s == "auto":
        # Pascal（算力 <7.0）无张量核且非张量 fp16 吞吐 = fp32 的 1/64 ⇒ 取 fp32
        cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
        s = "fp32" if cap[0] < 7 else "fp16"
    if s in ("fp32", "float32"):
        return torch.float32, "fp32"
    if s in ("fp16", "float16", "half"):
        return torch.float16, "fp16"
    raise ValueError(f"{op_name}: dtype 非法 {name!r}，可选 fp32/fp16/auto")


def _check_cuda(op_name):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"{op_name}: 需要 CUDA（GPU 推理），当前 "
                           f"torch.cuda.is_available()=False")


def _check_size(size, op_name):
    """内部推理边长：必须是 32 的倍数（ConvNeXt 4 级 /32 下采样）。"""
    n = int(size)
    if n < 128 or n > 1536 or n % 32:
        raise ValueError(f"{op_name}: size 非法 {size!r}（需 128~1536 且为 32 的倍数）")
    return n


def _register(key, obj, est_bytes, arch, op_name):
    """注册进 model_manager LRU（与整管同一淘汰/pin/unload 语义；flux.py 同款）。"""
    from app.main import models
    with models._lock:
        models._evict_if_needed(est_bytes)
        models._pipes[key] = obj
        models._archs[key] = arch
        models._sizes[key] = est_bytes
        models._last_used[key] = time.time()
    print(f"[{op_name}] registered '{key}' est={est_bytes / 1e9:.2f}GB", flush=True)
    return obj


def _cached(key):
    """取同进程常驻模型（touch LRU）；已被淘汰/未装载返回 None。"""
    try:
        from app.main import models
    except Exception:
        return None
    with models._lock:
        obj = models._pipes.get(key)
        if obj is not None:
            models._last_used[key] = time.time()
        return obj


def _residents(op_name):
    try:
        from app.main import models
        return "、".join(models.resident()) or "(无)"
    except Exception:
        return "(不可用)"


def _key_parts(model_ref, op_name):
    """`ddcolor:<variant>` → (键, 变体)；格式错时给可操作报错。"""
    key = str(model_ref or "").strip()
    if not key.startswith("ddcolor:"):
        raise ValueError(f"{op_name}: model 非法 {model_ref!r}（应为 colorize-load 输出的 "
                         f"'ddcolor:<variant>'）")
    variant = key.split(":", 1)[1]
    if variant not in VARIANTS:
        raise ValueError(f"{op_name}: model 里的变体非法 {variant!r}，可选 "
                         f"{sorted(VARIANTS)}")
    return key, variant


def _dumps(obj):
    """JSON 负载的信封编码：`@` + JSON（Studio 会把"像 JSON 的字符串"自动反序列化成
    map，注入节点 env 时变成 Go 的 `<map[string]interface {} Value>`，B3 真机复现）。"""
    return "@" + json.dumps(obj, ensure_ascii=False)


def _loads(text, default=None):
    """解信封（容忍无前缀的裸 JSON，便于手工在 UI 里填值）。"""
    s = str(text or "").strip()
    if s.startswith("@"):
        s = s[1:]
    if not s:
        return default
    return json.loads(s)


def _build_model(path, op_name):
    """按 config.json 构造 DDColor 并装载权重。

    权重键与架构不匹配时**大声失败**（dev-plan §50.7 的风险项：HF 只放权重不放代码，
    ModelScope 版 config 与 vendored 类的字段是对齐的，但需实测确认；缺真实权重键
    说明该变体需要 `basicsr/archs/ddcolor_arch.py` 那套 arch）。
    """
    import torch
    with open(os.path.join(path, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    DDColor = _ddclasses()["DDColor"]
    model = DDColor(
        encoder_name=cfg.get("encoder_name", "convnext-l"),
        decoder_name=cfg.get("decoder_name", "MultiScaleColorDecoder"),
        input_size=tuple(cfg.get("input_size", (512, 512))),
        num_output_channels=int(cfg.get("num_output_channels", 2)),
        last_norm=cfg.get("last_norm", "Spectral"),
        do_normalize=bool(cfg.get("do_normalize", False)),
        num_queries=int(cfg.get("num_queries", 100)),
        num_scales=int(cfg.get("num_scales", 3)),
        dec_layers=int(cfg.get("dec_layers", 9)),
    )
    ckpt = torch.load(os.path.join(path, "pytorch_model.bin"), map_location="cpu")
    if isinstance(ckpt, dict) and "params" in ckpt:      # 上游 load_checkpoint_state_dict
        ckpt = ckpt["params"]
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    crit = [k for k in missing if not k.endswith("num_batches_tracked")]
    if crit:
        raise RuntimeError(
            f"{op_name}: 权重与 vendored DDColor 架构不匹配：缺 {len(crit)} 个权重键"
            f"（例：{crit[:3]}）；该变体可能需 basicsr 版 arch（dev-plan §50.1.2）")
    info = "strict" if not (missing or unexpected) else (
        f"lenient(missing={len(missing)},unexpected={len(unexpected)})")
    print(f"[{op_name}] 权重装载 {info}"
          f"{'：' + str(sorted(missing)[:3]) if missing else ''}", flush=True)
    return model, info


def _ensure(key, variant, op_name):
    """取常驻模型；不在则按原 dtype 现装（交互式单跑也能用）。"""
    model = _cached(key)
    if model is not None:
        return model
    path = _variant_dir(COLORIZE_DIR, variant, op_name)
    dt_name = _DTYPES.get(key, "fp32")
    dt, dt_name = _dtype(dt_name, op_name)
    _check_cuda(op_name)
    t0 = time.time()
    model, info = _build_model(path, op_name)
    model = model.to(dtype=dt).to("cuda").eval()
    n = sum(p.numel() for p in model.parameters())
    _register(key, model, n * dt.itemsize, "ddcolor", op_name)
    _DTYPES[key] = dt_name
    _LOADINFO[key] = info
    print(f"[{op_name}] variant={variant} dtype={dt_name} params={n / 1e6:.1f}M "
          f"load={time.time() - t0:.1f}s 常驻={_residents(op_name)}", flush=True)
    return model


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------

def colorize_load(variant="modelscope", dtype="fp32", reload="false"):
    """装载 DDColor 变体 → 模型键 `ddcolor:<variant>`（进 model_manager LRU）。"""
    op_name = f"colorize.load@{OP_VERSION}"
    variant = str(variant or "modelscope").strip()
    key = f"ddcolor:{variant}"
    want_reload = str(reload).strip().lower() in ("1", "true", "yes", "on")
    if want_reload:
        from app.main import models
        models.unload(key)
        _DTYPES.pop(key, None)
    path = _variant_dir(COLORIZE_DIR, variant, op_name)      # 先快速失败
    dt, dt_name = _dtype(dtype, op_name)
    was = _cached(key)
    if was is not None and not want_reload and _DTYPES.get(key) != dt_name:
        from app.main import models                                     # 精度变了要重建
        models.unload(key)
        was = None
    model = _ensure(key, variant, op_name)
    n = sum(p.numel() for p in model.parameters())
    dt_used, dt_name_used = _dtype(_DTYPES.get(key, "fp32"), op_name)
    params_mb = int(n * dt_used.itemsize / 1024 ** 2)
    print(f"[{op_name}] {variant} params={n / 1e6:.1f}M dtype={dt_name_used} "
          f"vram≈{params_mb}MB 命中={was is not None} 常驻={_residents(op_name)}",
          flush=True)
    return {"model": key, "params_mb": params_mb, "vram_mb": params_mb,
            "info": _dumps({"variant": variant, "dtype": dt_name_used,
                            "params_mb": params_mb,
                            "load": _LOADINFO.get(key, "unknown"),
                            "cache_hit": was is not None})}


def _infer(model, pil, size, saturation):
    """`ColorizationPipeline.process` 的等价实现（RGB PIL 进出）。

    数值流程与上游逐行一致：
    RGB→BGR/[0,1] → 原图 L（最后拼回）→ 缩到 size → 灰图复制 L 成三通道 Lab→RGB
    → 网络预测 ab → 插值回原尺寸 → 拼原始 L → Lab→BGR→uint8。
    唯一差异：末步 clip 到 [0,1]（上游直接 astype(uint8)，越界会回绕）。
    """
    import cv2
    import numpy as np
    import torch
    import torch.nn.functional as F

    with torch.inference_mode():
        src = pil.convert("RGB")
        w, h = src.size
        arr = np.asarray(src)[:, :, ::-1].astype(np.float32) / 255.0     # RGB→BGR
        orig_lab = cv2.cvtColor(arr, cv2.COLOR_BGR2Lab)
        orig_l = orig_lab[:, :, :1]                                      # 原图亮度（结构不变）
        small = cv2.resize(arr, (size, size))
        small_l = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)[:, :, :1]
        gray_lab = np.concatenate((small_l, np.zeros_like(small_l),
                                   np.zeros_like(small_l)), axis=-1)
        gray_rgb = cv2.cvtColor(gray_lab, cv2.COLOR_LAB2RGB)
        t = torch.from_numpy(gray_rgb.transpose((2, 0, 1))).float().unsqueeze(0)
        t = t.to(device="cuda", dtype=next(model.parameters()).dtype)
        ab = model(t).float().cpu()                                      # (1,2,size,size)
        ab = F.interpolate(ab, size=(h, w))[0].float().numpy().transpose(1, 2, 0)
        chroma_in = float(np.sqrt(orig_lab[:, :, 1] ** 2
                                  + orig_lab[:, :, 2] ** 2).mean())
        if saturation != 1.0:
            ab = ab * float(saturation)
        # chroma 取**最终输出**（含 saturation），是落盘复核判据 A3 的直接观测值
        chroma = float(np.sqrt(ab[:, :, 0] ** 2 + ab[:, :, 1] ** 2).mean())
        out_lab = np.concatenate((orig_l, ab), axis=-1)
        out_bgr = cv2.cvtColor(out_lab, cv2.COLOR_LAB2BGR)
        out = (np.clip(out_bgr, 0.0, 1.0) * 255.0).round().astype(np.uint8)
        pil_out = Image.fromarray(out[:, :, ::-1])                       # BGR→RGB
    return pil_out, chroma, chroma_in


def colorize_apply(image, model, size=512, saturation=1.0):
    """上色：IMAGE + 模型键 → 上色 IMAGE（+ 色度指标，供落盘复核判据）。"""
    op_name = f"colorize.apply@{OP_VERSION}"
    key, variant = _key_parts(model, op_name)
    n = _check_size(size, op_name)
    sat = float(saturation)
    if not (0.0 < sat <= 4.0):
        raise ValueError(f"{op_name}: saturation 非法 {saturation!r}（需 0 < s <= 4）")
    net = _ensure(key, variant, op_name)
    t0 = time.time()
    out, chroma, chroma_in = _infer(net, image, n, sat)
    dt = time.time() - t0
    w, h = out.size
    print(f"[{op_name}] {variant} {image.size[0]}x{image.size[1]} → {w}x{h} "
          f"size={n} sat={sat} chroma={chroma:.1f}（输入 {chroma_in:.1f}）"
          f" {dt:.2f}s", flush=True)
    return {"image": out, "width": int(w), "height": int(h),
            "chroma": round(chroma, 2), "chroma_in": round(chroma_in, 2),
            "info": _dumps({"variant": variant, "dtype": _DTYPES.get(key, "fp32"),
                            "size": n, "saturation": sat, "chroma": round(chroma, 2),
                            "chroma_in": round(chroma_in, 2),
                            "elapsed_s": round(dt, 3)})}


def _ddclasses():
    """惰性构造 vendored DDColor 类（首次调用才定义；模块导入不 import torch）。

    **本函数体是 piddnad/DDColor @ master 的 5 个模块逐行内联**，顺序：
    `unet.py` → `convnext.py` → `transformer_utils.py` → `position_encoding.py`
    → `model.py`。仅去掉模块级 import（改由本函数顶部统一 import）与
    `from basicsr...` 交叉引用（同作用域内定义即可见），其余一字未改。
    """
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    import collections
    import math
    from enum import Enum
    from typing import Optional

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch import Tensor

    # ===================== basicsr/archs/ddcolor_arch_utils/unet.py =====================
    NormType = Enum('NormType', 'Batch BatchZero Weight Spectral')


    class Hook:
        feature = None

        def __init__(self, module):
            self.hook = module.register_forward_hook(self.hook_fn)

        def hook_fn(self, module, input, output):
            if isinstance(output, torch.Tensor):
                self.feature = output
            elif isinstance(output, collections.OrderedDict):
                self.feature = output['out']

        def remove(self):
            self.hook.remove()


    class SelfAttention(nn.Module):
        "Self attention layer for nd."

        def __init__(self, n_channels: int):
            super().__init__()
            self.query = conv1d(n_channels, n_channels // 8)
            self.key = conv1d(n_channels, n_channels // 8)
            self.value = conv1d(n_channels, n_channels)
            self.gamma = nn.Parameter(torch.tensor([0.]))

        def forward(self, x):
            #Notation from https://arxiv.org/pdf/1805.08318.pdf
            size = x.size()
            x = x.view(*size[:2], -1)
            f, g, h = self.query(x), self.key(x), self.value(x)
            beta = F.softmax(torch.bmm(f.permute(0, 2, 1).contiguous(), g), dim=1)
            o = self.gamma * torch.bmm(h, beta) + x
            return o.view(*size).contiguous()


    def batchnorm_2d(nf: int, norm_type: NormType = NormType.Batch):
        "A batchnorm2d layer with `nf` features initialized depending on `norm_type`."
        bn = nn.BatchNorm2d(nf)
        with torch.no_grad():
            bn.bias.fill_(1e-3)
            bn.weight.fill_(0. if norm_type == NormType.BatchZero else 1.)
        return bn


    def init_default(m: nn.Module, func=nn.init.kaiming_normal_) -> None:
        "Initialize `m` weights with `func` and set `bias` to 0."
        if func:
            if hasattr(m, 'weight'): func(m.weight)
            if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)
        return m


    def icnr(x, scale=2, init=nn.init.kaiming_normal_):
        "ICNR init of `x`, with `scale` and `init` function."
        ni, nf, h, w = x.shape
        ni2 = int(ni / (scale**2))
        k = init(torch.zeros([ni2, nf, h, w])).transpose(0, 1)
        k = k.contiguous().view(ni2, nf, -1)
        k = k.repeat(1, 1, scale**2)
        k = k.contiguous().view([nf, ni, h, w]).transpose(0, 1)
        x.data.copy_(k)


    def conv1d(ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False):
        "Create and initialize a `nn.Conv1d` layer with spectral normalization."
        conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
        nn.init.kaiming_normal_(conv.weight)
        if bias: conv.bias.data.zero_()
        return nn.utils.spectral_norm(conv)


    def custom_conv_layer(
        ni: int,
        nf: int,
        ks: int = 3,
        stride: int = 1,
        padding: int = None,
        bias: bool = None,
        is_1d: bool = False,
        norm_type=NormType.Batch,
        use_activ: bool = True,
        transpose: bool = False,
        init=nn.init.kaiming_normal_,
        self_attention: bool = False,
        extra_bn: bool = False,
    ):
        "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
        if padding is None:
            padding = (ks - 1) // 2 if not transpose else 0
        bn = norm_type in (NormType.Batch, NormType.BatchZero) or extra_bn == True
        if bias is None:
            bias = not bn
        conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
        conv = init_default(
            conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding),
            init,
        )

        if norm_type == NormType.Weight:
            conv = nn.utils.weight_norm(conv)
        elif norm_type == NormType.Spectral:
            conv = nn.utils.spectral_norm(conv)
        layers = [conv]
        if use_activ:
            layers.append(nn.ReLU(True))
        if bn:
            layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
        if self_attention:
            layers.append(SelfAttention(nf))
        return nn.Sequential(*layers)


    def conv_layer(ni: int,
                   nf: int,
                   ks: int = 3,
                   stride: int = 1,
                   padding: int = None,
                   bias: bool = None,
                   is_1d: bool = False,
                   norm_type=NormType.Batch,
                   use_activ: bool = True,
                   transpose: bool = False,
                   init=nn.init.kaiming_normal_,
                   self_attention: bool = False):
        "Create a sequence of convolutional (`ni` to `nf`), ReLU (if `use_activ`) and batchnorm (if `bn`) layers."
        if padding is None: padding = (ks - 1) // 2 if not transpose else 0
        bn = norm_type in (NormType.Batch, NormType.BatchZero)
        if bias is None: bias = not bn
        conv_func = nn.ConvTranspose2d if transpose else nn.Conv1d if is_1d else nn.Conv2d
        conv = init_default(conv_func(ni, nf, kernel_size=ks, bias=bias, stride=stride, padding=padding), init)
        if norm_type == NormType.Weight: conv = nn.utils.weight_norm(conv)
        elif norm_type == NormType.Spectral: conv = nn.utils.spectral_norm(conv)
        layers = [conv]
        if use_activ: layers.append(nn.ReLU(True))
        if bn: layers.append((nn.BatchNorm1d if is_1d else nn.BatchNorm2d)(nf))
        if self_attention: layers.append(SelfAttention(nf))
        return nn.Sequential(*layers)


    def _conv(ni: int, nf: int, ks: int = 3, stride: int = 1, **kwargs):
        return conv_layer(ni, nf, ks=ks, stride=stride, norm_type=NormType.Spectral, **kwargs)


    class CustomPixelShuffle_ICNR(nn.Module):
        "Upsample by `scale` from `ni` filters to `nf` (default `ni`), using `nn.PixelShuffle`, `icnr` init, and `weight_norm`."

        def __init__(self,
                     ni: int,
                     nf: int = None,
                     scale: int = 2,
                     blur: bool = True,
                     norm_type=NormType.Spectral,
                     extra_bn=False):
            super().__init__()
            self.conv = custom_conv_layer(
                ni, nf * (scale**2), ks=1, use_activ=False, norm_type=norm_type, extra_bn=extra_bn)
            icnr(self.conv[0].weight)
            self.shuf = nn.PixelShuffle(scale)
            self.do_blur = blur
            # Blurring over (h*w) kernel
            # "Super-Resolution using Convolutional Neural Networks without Any Checkerboard Artifacts"
            # - https://arxiv.org/abs/1806.02658
            self.pad = nn.ReplicationPad2d((1, 0, 1, 0))
            self.blur = nn.AvgPool2d(2, stride=1)
            self.relu = nn.ReLU(True)

        def forward(self, x):
            x = self.shuf(self.relu(self.conv(x)))
            return self.blur(self.pad(x)) if self.do_blur else x


    class UnetBlockWide(nn.Module):
        "A quasi-UNet block, using `PixelShuffle_ICNR upsampling`."

        def __init__(self,
                     up_in_c: int,
                     x_in_c: int,
                     n_out: int,
                     hook,
                     blur: bool = False,
                     self_attention: bool = False,
                     norm_type=NormType.Spectral):
            super().__init__()

            self.hook = hook
            up_out = n_out
            self.shuf = CustomPixelShuffle_ICNR(up_in_c, up_out, blur=blur, norm_type=norm_type, extra_bn=True)
            self.bn = batchnorm_2d(x_in_c)
            ni = up_out + x_in_c
            self.conv = custom_conv_layer(ni, n_out, norm_type=norm_type, self_attention=self_attention, extra_bn=True)
            self.relu = nn.ReLU()

        def forward(self, up_in):
            s = self.hook.feature
            up_out = self.shuf(up_in)
            cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
            return self.conv(cat_x)

    # ===================== basicsr/archs/ddcolor_arch_utils/convnext.py =====================
    # Copyright (c) Meta Platforms, Inc. and affiliates.

    # All rights reserved.

    # This source code is licensed under the license found in the
    # LICENSE file in the root directory of this source tree.



    # ---- Optional dependency: timm ----
    # This file only needs two small helpers from timm: `trunc_normal_` and `DropPath`.
    # To reduce inference dependencies, we provide a pure-PyTorch fallback implementation.
    try:
        from timm.layers import trunc_normal_, DropPath  # type: ignore
    except Exception:
        import math

        def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
            """Fills the input Tensor with values drawn from a truncated normal distribution.

            Fallback implementation when timm is not available.
            """
            # Prefer PyTorch built-in if present
            if hasattr(torch.nn.init, "trunc_normal_"):
                return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)

            # Based on PyTorch's internal implementation pattern
            def norm_cdf(x):
                return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

            with torch.no_grad():
                l = norm_cdf((a - mean) / std)
                u = norm_cdf((b - mean) / std)

                tensor.uniform_(2 * l - 1, 2 * u - 1)
                tensor.erfinv_()

                tensor.mul_(std * math.sqrt(2.0))
                tensor.add_(mean)
                tensor.clamp_(min=a, max=b)
                return tensor


        class DropPath(nn.Module):
            """Stochastic Depth per sample (when applied in main path of residual blocks)."""

            def __init__(self, drop_prob: float = 0.0):
                super().__init__()
                self.drop_prob = float(drop_prob)

            def forward(self, x):
                if self.drop_prob == 0.0 or not self.training:
                    return x
                keep_prob = 1.0 - self.drop_prob
                shape = (x.shape[0],) + (1,) * (x.ndim - 1)
                random_tensor = keep_prob + torch.rand(
                    shape, dtype=x.dtype, device=x.device
                )
                random_tensor.floor_()
                return x.div(keep_prob) * random_tensor

    class Block(nn.Module):
        r""" ConvNeXt Block. There are two equivalent implementations:
        (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
        (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
        We use (2) as we find it slightly faster in PyTorch

        Args:
            dim (int): Number of input channels.
            drop_path (float): Stochastic depth rate. Default: 0.0
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        """
        def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
            super().__init__()
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
            self.norm = LayerNorm(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(4 * dim, dim)
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                        requires_grad=True) if layer_scale_init_value > 0 else None
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(self, x):
            input = x
            x = self.dwconv(x)
            x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
            x = self.norm(x)
            x = self.pwconv1(x)
            x = self.act(x)
            x = self.pwconv2(x)
            if self.gamma is not None:
                x = self.gamma * x
            x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

            x = input + self.drop_path(x)
            return x

    class ConvNeXt(nn.Module):
        r""" ConvNeXt
            A PyTorch impl of : `A ConvNet for the 2020s`  -
              https://arxiv.org/pdf/2201.03545.pdf
        Args:
            in_chans (int): Number of input image channels. Default: 3
            num_classes (int): Number of classes for classification head. Default: 1000
            depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
            dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
            drop_path_rate (float): Stochastic depth rate. Default: 0.
            layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
            head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        """
        def __init__(self, in_chans=3, num_classes=1000,
                     depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0.,
                     layer_scale_init_value=1e-6, head_init_scale=1.,
                     ):
            super().__init__()

            self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
            stem = nn.Sequential(
                nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
            )
            self.downsample_layers.append(stem)
            for i in range(3):
                downsample_layer = nn.Sequential(
                        LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                        nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                )
                self.downsample_layers.append(downsample_layer)

            self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
            dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
            cur = 0
            for i in range(4):
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                    layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
                self.stages.append(stage)
                cur += depths[i]

            # add norm layers for each output
            out_indices = (0, 1, 2, 3)
            for i in out_indices:
                layer = LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
                # layer = nn.Identity()
                layer_name = f'norm{i}'
                self.add_module(layer_name, layer)

            self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
            # self.head_cls = nn.Linear(dims[-1], 4)

            self.apply(self._init_weights)
            # self.head_cls.weight.data.mul_(head_init_scale)
            # self.head_cls.bias.data.mul_(head_init_scale)

        def _init_weights(self, m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

        def forward_features(self, x):
            for i in range(4):
                x = self.downsample_layers[i](x)
                x = self.stages[i](x)

                # add extra norm
                norm_layer = getattr(self, f'norm{i}')
                # x = norm_layer(x)
                norm_layer(x)

            return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

        def forward(self, x):
            x = self.forward_features(x)
            # x = self.head_cls(x)
            return x

    class LayerNorm(nn.Module):
        r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
        The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
        shape (batch_size, height, width, channels) while channels_first corresponds to inputs
        with shape (batch_size, channels, height, width).
        """
        def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
            self.eps = eps
            self.data_format = data_format
            if self.data_format not in ["channels_last", "channels_first"]:
                raise NotImplementedError
            self.normalized_shape = (normalized_shape, )

        def forward(self, x):
            if self.data_format == "channels_last":  # B H W C
                return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            elif self.data_format == "channels_first":  # B C H W
                u = x.mean(1, keepdim=True)
                s = (x - u).pow(2).mean(1, keepdim=True)
                x = (x - u) / torch.sqrt(s + self.eps)
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
                return x

    # ===================== basicsr/archs/ddcolor_arch_utils/transformer_utils.py =====================
    class SelfAttentionLayer(nn.Module):

        def __init__(self, d_model, nhead, dropout=0.0,
                     activation="relu", normalize_before=False):
            super().__init__()
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

            self.activation = _get_activation_fn(activation)
            self.normalize_before = normalize_before

            self._reset_parameters()

        def _reset_parameters(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def with_pos_embed(self, tensor, pos: Optional[Tensor]):
            return tensor if pos is None else tensor + pos

        def forward_post(self, tgt,
                         tgt_mask: Optional[Tensor] = None,
                         tgt_key_padding_mask: Optional[Tensor] = None,
                         query_pos: Optional[Tensor] = None):
            q = k = self.with_pos_embed(tgt, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)

            return tgt

        def forward_pre(self, tgt,
                        tgt_mask: Optional[Tensor] = None,
                        tgt_key_padding_mask: Optional[Tensor] = None,
                        query_pos: Optional[Tensor] = None):
            tgt2 = self.norm(tgt)
            q = k = self.with_pos_embed(tgt2, query_pos)
            tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)

            return tgt

        def forward(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
            if self.normalize_before:
                return self.forward_pre(tgt, tgt_mask,
                                        tgt_key_padding_mask, query_pos)
            return self.forward_post(tgt, tgt_mask,
                                     tgt_key_padding_mask, query_pos)


    class CrossAttentionLayer(nn.Module):

        def __init__(self, d_model, nhead, dropout=0.0,
                     activation="relu", normalize_before=False):
            super().__init__()
            self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

            self.norm = nn.LayerNorm(d_model)
            self.dropout = nn.Dropout(dropout)

            self.activation = _get_activation_fn(activation)
            self.normalize_before = normalize_before

            self._reset_parameters()

        def _reset_parameters(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def with_pos_embed(self, tensor, pos: Optional[Tensor]):
            return tensor if pos is None else tensor + pos

        def forward_post(self, tgt, memory,
                         memory_mask: Optional[Tensor] = None,
                         memory_key_padding_mask: Optional[Tensor] = None,
                         pos: Optional[Tensor] = None,
                         query_pos: Optional[Tensor] = None):
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)

            return tgt

        def forward_pre(self, tgt, memory,
                        memory_mask: Optional[Tensor] = None,
                        memory_key_padding_mask: Optional[Tensor] = None,
                        pos: Optional[Tensor] = None,
                        query_pos: Optional[Tensor] = None):
            tgt2 = self.norm(tgt)
            tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                       key=self.with_pos_embed(memory, pos),
                                       value=memory, attn_mask=memory_mask,
                                       key_padding_mask=memory_key_padding_mask)[0]
            tgt = tgt + self.dropout(tgt2)

            return tgt

        def forward(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
            if self.normalize_before:
                return self.forward_pre(tgt, memory, memory_mask,
                                        memory_key_padding_mask, pos, query_pos)
            return self.forward_post(tgt, memory, memory_mask,
                                     memory_key_padding_mask, pos, query_pos)


    class FFNLayer(nn.Module):

        def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                     activation="relu", normalize_before=False):
            super().__init__()
            # Implementation of Feedforward model
            self.linear1 = nn.Linear(d_model, dim_feedforward)
            self.dropout = nn.Dropout(dropout)
            self.linear2 = nn.Linear(dim_feedforward, d_model)

            self.norm = nn.LayerNorm(d_model)

            self.activation = _get_activation_fn(activation)
            self.normalize_before = normalize_before

            self._reset_parameters()

        def _reset_parameters(self):
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        def with_pos_embed(self, tensor, pos: Optional[Tensor]):
            return tensor if pos is None else tensor + pos

        def forward_post(self, tgt):
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
            tgt = tgt + self.dropout(tgt2)
            tgt = self.norm(tgt)
            return tgt

        def forward_pre(self, tgt):
            tgt2 = self.norm(tgt)
            tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
            tgt = tgt + self.dropout(tgt2)
            return tgt

        def forward(self, tgt):
            if self.normalize_before:
                return self.forward_pre(tgt)
            return self.forward_post(tgt)


    def _get_activation_fn(activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


    class MLP(nn.Module):
        """ Very simple multi-layer perceptron (also called FFN)"""

        def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
            super().__init__()
            self.num_layers = num_layers
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
            return x

    # ===================== basicsr/archs/ddcolor_arch_utils/position_encoding.py =====================
    # Copyright (c) Facebook, Inc. and its affiliates.
    # Modified from: https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """
    Various positional encodings for the transformer.
    """



    class PositionEmbeddingSine(nn.Module):
        """
        This is a more standard version of the position embedding, very similar to the one
        used by the Attention is all you need paper, generalized to work on images.
        """

        def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
            super().__init__()
            self.num_pos_feats = num_pos_feats
            self.temperature = temperature
            self.normalize = normalize
            if scale is not None and normalize is False:
                raise ValueError("normalize should be True if scale is passed")
            if scale is None:
                scale = 2 * math.pi
            self.scale = scale

        def forward(self, x, mask=None):
            if mask is None:
                mask = torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool)
            not_mask = ~mask
            y_embed = not_mask.cumsum(1, dtype=torch.float32)
            x_embed = not_mask.cumsum(2, dtype=torch.float32)
            if self.normalize:
                eps = 1e-6
                y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
                x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

            dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
            dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t
            pos_x = torch.stack(
                (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos_y = torch.stack(
                (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
            ).flatten(3)
            pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
            return pos

    # ===================== ddcolor/model.py =====================
    class DDColor(nn.Module):
        def __init__(
            self,
            encoder_name='convnext-l',
            decoder_name='MultiScaleColorDecoder',
            num_input_channels=3,
            input_size=(256, 256),
            nf=512,
            num_output_channels=3,
            last_norm='Weight',
            do_normalize=False,
            num_queries=256,
            num_scales=3,
            dec_layers=9,
        ):
            super().__init__()

            self.encoder = ImageEncoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'])
            self.encoder.eval()
            test_input = torch.randn(1, num_input_channels, *input_size)

            with torch.no_grad():
                self.encoder(test_input)

            self.decoder = DuelDecoder(
                self.encoder.hooks,
                nf=nf,
                last_norm=last_norm,
                num_queries=num_queries,
                num_scales=num_scales,
                dec_layers=dec_layers,
                decoder_name=decoder_name
            )

            self.refine_net = nn.Sequential(
                custom_conv_layer(num_queries + 3, num_output_channels, ks=1, use_activ=False, norm_type=NormType.Spectral)
            )

            self.do_normalize = do_normalize
            self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def normalize(self, img):
            return (img - self.mean) / self.std

        def denormalize(self, img):
            return img * self.std + self.mean

        def forward(self, x):
            if x.shape[1] == 3:
                x = self.normalize(x)

            self.encoder(x)
            out_feat = self.decoder()
            coarse_input = torch.cat([out_feat, x], dim=1)
            out = self.refine_net(coarse_input)

            if self.do_normalize:
                out = self.denormalize(out)
            return out


    class ImageEncoder(nn.Module):
        def __init__(self, encoder_name, hook_names):
            super().__init__()

            assert encoder_name == 'convnext-t' or encoder_name == 'convnext-l'
            if encoder_name == 'convnext-t':
                self.arch = ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768])
            elif encoder_name == 'convnext-l':
                self.arch = ConvNeXt(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536])
            else:
                raise NotImplementedError

            self.encoder_name = encoder_name
            self.hook_names = hook_names
            self.hooks = self.setup_hooks()

        def setup_hooks(self):
            hooks = [Hook(self.arch._modules[name]) for name in self.hook_names]
            return hooks

        def forward(self, x):
            return self.arch(x)


    class DuelDecoder(nn.Module):
        def __init__(
            self,
            hooks,
            nf=512,
            blur=True,
            last_norm='Weight',
            num_queries=256,
            num_scales=3,
            dec_layers=9,
            decoder_name='MultiScaleColorDecoder',
        ):
            super().__init__()
            self.hooks = hooks
            self.nf = nf
            self.blur = blur
            self.last_norm = getattr(NormType, last_norm)
            self.decoder_name = decoder_name

            self.layers = self.make_layers()
            embed_dim = nf // 2
            self.last_shuf = CustomPixelShuffle_ICNR(embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)

            assert decoder_name == 'MultiScaleColorDecoder'
            self.color_decoder = MultiScaleColorDecoder(
                in_channels=[512, 512, 256],
                num_queries=num_queries,
                num_scales=num_scales,
                dec_layers=dec_layers,
            )

        def make_layers(self):
            decoder_layers = []
            in_c = self.hooks[-1].feature.shape[1]
            out_c = self.nf

            setup_hooks = self.hooks[-2::-1]
            for layer_index, hook in enumerate(setup_hooks):
                feature_c = hook.feature.shape[1]
                if layer_index == len(setup_hooks) - 1:
                    out_c = out_c // 2
                decoder_layers.append(
                    UnetBlockWide(
                        in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral))
                in_c = out_c

            return nn.Sequential(*decoder_layers)

        def forward(self):
            encode_feat = self.hooks[-1].feature
            out0 = self.layers[0](encode_feat)
            out1 = self.layers[1](out0)
            out2 = self.layers[2](out1)
            out3 = self.last_shuf(out2)

            return self.color_decoder([out0, out1, out2], out3)


    class MultiScaleColorDecoder(nn.Module):
        def __init__(
            self,
            in_channels,
            hidden_dim=256,
            num_queries=100,
            nheads=8,
            dim_feedforward=2048,
            dec_layers=9,
            pre_norm=False,
            color_embed_dim=256,
            enforce_input_project=True,
            num_scales=3,
        ):
            super().__init__()

            self.hidden_dim = hidden_dim
            self.num_queries = num_queries
            self.num_layers = dec_layers
            self.num_feature_levels = num_scales

            # Positional encoding layer
            self.pe_layer = PositionEmbeddingSine(hidden_dim // 2, normalize=True)

            # Learnable query features and embeddings
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

            # Learnable level embeddings
            self.level_embed = nn.Embedding(num_scales, hidden_dim)

            # Input projection layers
            self.input_proj = nn.ModuleList(
                [self._make_input_proj(in_ch, hidden_dim, enforce_input_project) for in_ch in in_channels]
            )

            # Transformer layers
            self.transformer_self_attention_layers = nn.ModuleList()
            self.transformer_cross_attention_layers = nn.ModuleList()
            self.transformer_ffn_layers = nn.ModuleList()

            for _ in range(dec_layers):
                self.transformer_self_attention_layers.append(
                    SelfAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
                self.transformer_cross_attention_layers.append(
                    CrossAttentionLayer(
                        d_model=hidden_dim,
                        nhead=nheads,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )
                self.transformer_ffn_layers.append(
                    FFNLayer(
                        d_model=hidden_dim,
                        dim_feedforward=dim_feedforward,
                        dropout=0.0,
                        normalize_before=pre_norm,
                    )
                )

            # Layer normalization for the decoder output
            self.decoder_norm = nn.LayerNorm(hidden_dim)

            # Output embedding layer
            self.color_embed = MLP(hidden_dim, hidden_dim, color_embed_dim, 3)

        def forward(self, x, img_features):
            assert len(x) == self.num_feature_levels

            src, pos = self._get_src_and_pos(x)

            bs = src[0].shape[1]

            # Prepare query embeddings (QxNxC)
            query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
            output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

            for i in range(self.num_layers):
                level_index = i % self.num_feature_levels
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=None,
                    memory_key_padding_mask=None,
                    pos=pos[level_index], query_pos=query_embed
                )
                output = self.transformer_self_attention_layers[i](
                    output, tgt_mask=None,
                    tgt_key_padding_mask=None,
                    query_pos=query_embed
                )
                # FFN
                output = self.transformer_ffn_layers[i](
                    output
                )

            decoder_output = self.decoder_norm(output).transpose(0, 1)
            color_embed = self.color_embed(decoder_output)

            out = torch.einsum("bqc,bchw->bqhw", color_embed, img_features)

            return out

        def _make_input_proj(self, in_ch, hidden_dim, enforce):
            if in_ch != hidden_dim or enforce:
                proj = nn.Conv2d(in_ch, hidden_dim, kernel_size=1)
                nn.init.kaiming_uniform_(proj.weight, a=1)
                if proj.bias is not None:
                    nn.init.constant_(proj.bias, 0)
                return proj
            return nn.Sequential()

        def _get_src_and_pos(self, x):
            src, pos = [], []
            for i, feature in enumerate(x):
                pos.append(self.pe_layer(feature).flatten(2).permute(2, 0, 1))  # flatten NxCxHxW to HWxNxC
                src.append((self.input_proj[i](feature).flatten(2) + self.level_embed.weight[i][None, :, None]).permute(2, 0, 1))
            return src, pos

    _CLASSES = {"DDColor": DDColor, "ImageEncoder": ImageEncoder,
                "DuelDecoder": DuelDecoder,
                "MultiScaleColorDecoder": MultiScaleColorDecoder}
    return _CLASSES



def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "colorize.load",
        inputs={"variant": "STRING", "dtype": "STRING", "reload": "STRING"},
        outputs={"model": "STRING", "params_mb": "INT", "vram_mb": "INT",
                 "info": "STRING"},
        description="装载 DDColor 上色模型（variant=modelscope|paper|artistic|paper_tiny；"
                    "权重在 COLORIZE_MODELS_DIR/<variant>/，需 config.json + "
                    "pytorch_model.bin）→ 模型键 'ddcolor:<variant>'，注册进 "
                    "model_manager LRU（可 model-unload / LRU 淘汰）。dtype 默认 fp32"
                    "（Pascal fp16 吞吐 1/64）；reload=true 强制重装"
    )(colorize_load)
    registry.register(
        "colorize.apply",
        inputs={"image": "IMAGE", "model": "STRING", "size": "INT",
                "saturation": "FLOAT"},
        outputs={"image": "IMAGE", "width": "INT", "height": "INT",
                 "chroma": "FLOAT", "chroma_in": "FLOAT", "info": "STRING"},
        description="DDColor 上色（G8）：黑白/褪色图 → 彩色图。**保留原图 L 通道**、"
                    "只预测 ab ⇒ 结构/亮度逐像素不变；size 为内部推理边长（默认 512，"
                    "需 32 的倍数）；saturation 在 Lab 空间缩放色度（默认 1.0）。"
                    "chroma/chroma_in 是输出/输入的色度均值（落盘复核判据）"
    )(colorize_apply)
