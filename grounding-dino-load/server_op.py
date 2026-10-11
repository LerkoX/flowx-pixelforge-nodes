"""SAM1 / GroundingDINO 分割算子族（G2 / B3 批次，dev-plan §四十九）。

算子：
- `seg.sam_load`        ：装载 SAM1 变体（vit_h/l/b）→ 模型键（STRING）+ 参数量/显存
- `seg.sam`             ：IMAGE + points/box 提示 → 前景蒙版（灰度 IMAGE）+ 候选 + 叠加图
- `seg.sam_auto`        ：IMAGE → 自动全图分割（编号网格图 + 选中并集 + 各蒙版对象 ID）
- `seg.grounding_load`  ：装载 GroundingDINO（tiny/base）→ 模型键
- `seg.grounding_detect`：文本提示 → 框/标签/分数（`sam_model` 非空则再把框喂 SAM 出**像素级**蒙版）

模型口径（全部本地目录，零网络访问）：
- SAM1：`facebook/sam-vit-huge|large|base` → `SAM_MODELS_DIR/<variant>/`
  （`config.json` + `preprocessor_config.json` + `model.safetensors`）
- GroundingDINO：`IDEA-Research/grounding-dino-tiny|base` → `GROUNDING_MODELS_DIR/<variant>/`
- 依赖：仅 `transformers`（4.44.2 已含 `SamModel`/`GroundingDinoForObjectDetection`，实测见 dev-plan §49.2）
  + `timm`（GroundingDINO 的 Swin backbone，B2 已装）⇒ **本族零新增依赖**

两个关键设计（dev-plan §49.1）：
1. **注册进 model_manager LRU**（复用 `flux.py` 的 `_register` 写法）⇒ SAM/GD 模型自动获得
   淘汰 / pin / `model-unload` / `/models` 常驻列表全套语义，键为 `sam:<variant>` / `gd:<variant>`。
2. **dtype 默认 fp32**：Pascal（GP104, sm_61）非张量 fp16 吞吐 = fp32 的 **1/64**（dev-plan §49.2），
   `auto` 在算力 <7.0 的卡上同样取 fp32。换卡可用 `dtype=fp16` 显式指定。

交互式点选性能：SAM 的 image embedding 按（模型键 + 图片内容哈希）缓存一次，
同一张图的连续点选查询不重复编码（点选编辑器流畅的前提，`get_image_embeddings` 实测可用）。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "f18314e7cf62"
OP_FAMILY = "seg"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

import hashlib
import json
import os
import time

from PIL import Image, ImageDraw

SAM_DIR = os.environ.get("SAM_MODELS_DIR", "/models/sam")
GROUNDING_DIR = os.environ.get("GROUNDING_MODELS_DIR", "/models/grounding")

# 变体 → 上游 HF 仓库（只用于报错时给下载指引）
SAM_VARIANTS = {"vit_h": "facebook/sam-vit-huge",
                "vit_l": "facebook/sam-vit-large",
                "vit_b": "facebook/sam-vit-base"}
GROUNDING_VARIANTS = {"tiny": "IDEA-Research/grounding-dino-tiny",
                      "base": "IDEA-Research/grounding-dino-base"}

# 变体目录必备文件（缺哪个人话点名）
REQUIRED_FILES = ("config.json", "preprocessor_config.json", "model.safetensors")

_DTYPES = {}   # 模型键 -> dtype 字符串（淘汰后按原精度重建）
_PROCS = {}    # 模型键 -> processor（同进程常驻，极小）
_EMB = {}      # (模型键, 图片哈希) -> (embeddings, original_sizes, reshaped_input_sizes)
_EMB_ORDER = []  # 编码缓存淘汰顺序（只留最后 3 张，交互切图时不涨内存）


# ---------------------------------------------------------------------------
# 目录 / dtype / 显存记账
# ---------------------------------------------------------------------------

def _variant_dir(base, variant, allowed, op_name):
    """变体目录定位 + 必备文件检查（纯 os 操作，先快速失败）。"""
    if variant not in allowed:
        raise ValueError(f"{op_name}: variant 非法 {variant!r}，可选 "
                         f"{sorted(allowed)}")
    path = os.path.join(base, variant)
    if not os.path.isdir(path):
        repo = allowed[variant]
        raise FileNotFoundError(
            f"{op_name}: 变体目录不存在 {path}（下载 {repo} 到该目录；"
            f"环境变量指向 {base}）")
    missing = [f for f in REQUIRED_FILES
               if not os.path.isfile(os.path.join(path, f))]
    if missing:
        repo = allowed[variant]
        raise FileNotFoundError(
            f"{op_name}: 变体 {variant} 权重缺失 {path}/{'、'.join(missing)}"
            f"（需 config.json / preprocessor_config.json / model.safetensors，"
            f"来自 {repo}；下载见 nodes/README.md 权重清单）")
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


def _est_bytes(path, dtype_name):
    """按目标 dtype 精确估算权重显存（读 safetensors header；失败退文件大小）。"""
    p = os.path.join(path, "model.safetensors")
    try:
        from app import vram
        n = vram.safetensors_bytes(p, dtype_name)
    except Exception:
        n = None
    return int(n or os.path.getsize(p))


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


def _check_cuda(op_name):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"{op_name}: 需要 CUDA（GPU 推理），当前 "
                           f"torch.cuda.is_available()=False")


# ---------------------------------------------------------------------------
# 装载
# ---------------------------------------------------------------------------

def _sam_proc(key, variant, op_name):
    """SAM processor 进程级缓存（与 model 分开缓存）。

    必须容忍「模型已被**旧版族实例**装载并常驻」：F1 版本化后插件换名重传，
    新模块的 `_PROCS` 是空的，而 model_manager 里 `sam:<variant>` 仍在
    ⇒ `_cached` 命中时若不补 processor，`seg.sam` 会 KeyError（B3 真机复现：
    `RuntimeError: job ... failed: 'sam:vit_l'`）。
    """
    proc = _PROCS.get(key)
    if proc is None:
        from transformers import SamProcessor
        path = _variant_dir(SAM_DIR, variant, SAM_VARIANTS, op_name)
        proc = SamProcessor.from_pretrained(path, local_files_only=True)
        _PROCS[key] = proc
    return proc


def _gd_proc(key, variant, op_name):
    """GroundingDINO processor 进程级缓存（同上：跨版本收养要补齐 processor）。"""
    proc = _PROCS.get(key)
    if proc is None:
        from transformers import AutoProcessor
        path = _variant_dir(GROUNDING_DIR, variant, GROUNDING_VARIANTS, op_name)
        proc = AutoProcessor.from_pretrained(path, local_files_only=True)
        _PROCS[key] = proc
    return proc


def _load_sam(variant, dtype_name, op_name):
    key = f"sam:{variant}"
    obj = _cached(key)
    if obj is not None:
        _sam_proc(key, variant, op_name)  # 旧版实例装的模型：补 processor
        return obj
    _check_cuda(op_name)
    path = _variant_dir(SAM_DIR, variant, SAM_VARIANTS, op_name)
    import torch
    from transformers import SamModel, SamProcessor
    dt, dt_name = _dtype(dtype_name, op_name)
    model = SamModel.from_pretrained(path, torch_dtype=dt,
                                     local_files_only=True)
    model.eval().to("cuda")
    _PROCS[key] = SamProcessor.from_pretrained(path, local_files_only=True)
    _DTYPES[key] = dt_name
    return _register(key, model, _est_bytes(path, dt_name),
                     f"SamModel({variant}, {dt_name})", op_name)


def _load_grounding(variant, dtype_name, op_name):
    key = f"gd:{variant}"
    obj = _cached(key)
    if obj is not None:
        _gd_proc(key, variant, op_name)
        return obj
    _check_cuda(op_name)
    path = _variant_dir(GROUNDING_DIR, variant, GROUNDING_VARIANTS, op_name)
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    dt, dt_name = _dtype(dtype_name, op_name)
    proc = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        path, torch_dtype=dt, local_files_only=True)
    model.eval().to("cuda")
    _PROCS[key] = proc
    _DTYPES[key] = dt_name
    return _register(key, model, _est_bytes(path, dt_name),
                     f"GroundingDinoForObjectDetection({variant}, {dt_name})",
                     op_name)


def _parse_key(model, kind, allowed, op_name):
    """模型键解析：接受 'sam:vit_l' 或裸 'vit_l'。"""
    s = str(model or "").strip()
    if not s:
        raise ValueError(f"{op_name}: model 不能为空（如 "
                         f"'{kind}:{sorted(allowed)[0]}'）")
    if s.startswith(f"{kind}:"):
        s = s.split(":", 1)[1]
    return s


# ---------------------------------------------------------------------------
# 提示解析 / 绘制
# ---------------------------------------------------------------------------

def _dumps(obj):
    """JSON 负载的信封编码：`@` + JSON。

    为什么要加前缀：Studio 的参数绑定对"看起来是 JSON 的字符串值"会**自动反序列化**
    成 map，注入节点 env 时变成 Go 的 `<map[string]interface {} Value>`（B3 真机复现：
    tiled.apply 收到该字面串 → `plan 不是合法 JSON（char 0）`）。加一个非 JSON 字符
    前缀即可让它保持字符串原样传递；本族算子/前端都用 _loads() 解信封。
    """
    return "@" + json.dumps(obj, ensure_ascii=False)


def _loads(text, default=None):
    """解信封（容忍无前缀的裸 JSON，便于手工在 UI 里填值）。"""
    s = str(text or "").strip()
    if s.startswith("@"):
        s = s[1:]
    if not s:
        return default
    return json.loads(s)


def _parse_points(s, op_name):

    """'x,y,label;x,y,label' → ([x,y], [label])；label 缺省 1（正样本），0=负样本。"""
    pts, labels = [], []
    text = str(s or "").replace("\n", ";").replace("|", ";")
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p for p in chunk.replace(" ", ",").split(",") if p != ""]
        if len(parts) < 2:
            raise ValueError(f"{op_name}: points 片段非法 {chunk!r}"
                             f"（格式 'x,y[,label]'，多组用 ; 分隔）")
        x, y = float(parts[0]), float(parts[1])
        lab = int(float(parts[2])) if len(parts) > 2 else 1
        if lab not in (0, 1):
            raise ValueError(f"{op_name}: points label 只能 0/1，got {lab}")
        pts.append([x, y])
        labels.append(lab)
    return pts, labels


def _parse_box(s, op_name):
    """'x1,y1,x2,y2' → [x1,y1,x2,y2]；空返回 None。"""
    text = str(s or "").replace(" ", ",")
    parts = [p for p in text.split(",") if p != ""]
    if not parts:
        return None
    if len(parts) != 4:
        raise ValueError(f"{op_name}: box 需要 4 个数 'x1,y1,x2,y2'，got {s!r}")
    x1, y1, x2, y2 = (float(v) for v in parts)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{op_name}: box 需 x2>x1 且 y2>y1，got {s!r}")
    return [x1, y1, x2, y2]


def _as_single(m):
    """单个蒙版张量 → 2D bool numpy（容忍 (1,H,W)/(B,1,H,W) 等多余维）。"""
    import numpy as np
    arr = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    arr = np.asarray(arr)
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (3, 4):  # 极端情况：被当成多通道
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"蒙版维度异常 {arr.shape}（期望 H×W）")
    return arr > 0


def _norm_masks(res, op_name="seg"):
    """post_process_masks 结果 → (n, H, W) bool numpy。

    不同 transformers 版本/分支有的返回 list[per-image]、有的直接返回
    (batch, n, H, W)，这里统一归一，不依赖具体形态。
    """
    import numpy as np
    m = res[0] if isinstance(res, (list, tuple)) else res
    arr = m.detach().cpu().numpy() if hasattr(m, "detach") else np.asarray(m)
    arr = np.asarray(arr)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[None]
    if arr.ndim != 3:
        raise ValueError(f"{op_name}: 蒙版维度异常 {arr.shape}")
    return arr > 0


def _mask_to_pil(m):
    """单个蒙版（张量/数组，2D 或带单维） → 8bit 灰度 PIL（0/255）。"""
    return Image.fromarray(_as_single(m).astype("uint8") * 255, "L")


def _annotated(image, box, pts, labels, mask_pil, overlay):
    """原图叠提示（框/点）+ 蒙版半透明覆盖。"""
    base = image.convert("RGB")
    out = base
    if mask_pil is not None and float(overlay) > 0:
        tint = Image.new("RGB", base.size, (0, 200, 255))
        blend = Image.blend(base, tint, max(0.0, min(1.0, float(overlay))))
        out = Image.composite(blend, base, mask_pil.convert("L"))
    d = ImageDraw.Draw(out)
    if box is not None:
        d.rectangle(box, outline=(255, 64, 64), width=3)
    r = max(4, round(min(out.size) * 0.012))
    for (x, y), lab in zip(pts or [], labels or []):
        color = (0, 255, 64) if int(lab) == 1 else (255, 64, 64)
        d.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))
    return out


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------

def sam_load(variant="vit_l", dtype="fp32"):
    """装载 SAM1 变体 → 模型键（STRING）。返回键名供 seg.sam/seg.sam_auto 复用。"""
    op_name = f"seg.sam_load@{OP_VERSION}"
    variant = str(variant or "vit_l").strip().lower()
    path = _variant_dir(SAM_DIR, variant, SAM_VARIANTS, op_name)  # 先快速失败
    model = _load_sam(variant, dtype, op_name)
    n = sum(p.numel() for p in model.parameters())
    from app.main import models
    key = f"sam:{variant}"
    vram_mb = int(models._sizes.get(key, 0) / 1024 ** 2)
    print(f"[{op_name}] {variant} params={n / 1e6:.1f}M vram≈{vram_mb}MB "
          f"dtype={_DTYPES.get(key)} 常驻={_residents(op_name)}", flush=True)
    return {"model": key, "params_mb": int(n * 4 / 1024 ** 2), "vram_mb": vram_mb}


def _embeddings(key, model, proc, image):
    """SAM image embedding（按图片内容缓存）：同图连续点选不重复编码。"""
    h = hashlib.sha256()
    h.update(key.encode())
    h.update(image.convert("RGB").tobytes())
    h.update(str(image.size).encode())
    ih = h.hexdigest()
    cache_key = (key, ih)
    hit = _EMB.get(cache_key)
    if hit is not None:
        return hit
    import torch
    with torch.no_grad():
        px = proc(image, return_tensors="pt")["pixel_values"]
        px = px.to("cuda").to(next(model.parameters()).dtype)
        emb = model.get_image_embeddings(px)
    inputs = proc(image, return_tensors="pt")
    _EMB[cache_key] = (emb, inputs["original_sizes"], inputs["reshaped_input_sizes"])
    _EMB_ORDER.append(cache_key)
    while len(_EMB_ORDER) > 3:  # 只留 3 张图的编码（切图不涨内存）
        _EMB.pop(_EMB_ORDER.pop(0), None)
    return _EMB[cache_key]


def sam(image, model="sam:vit_l", points="", box="", multimask=True, pick=0,
        overlay=0.5):
    """SAM1 点/框提示分割：IMAGE + points/box → 灰度蒙版 + 候选 + 叠加预览。

    - points：'x,y,label;x,y,label'（label 1=正样本 0=负样本；缺省 1）
    - box：'x1,y1,x2,y2'（可与 points 同时给）
    - multimask：True=出 3 个候选（Ambiguity 消解），False=只出 1 个
    - pick：多候选时选第几个（0/1/2）
    """
    op_name = f"seg.sam@{OP_VERSION}"
    if not isinstance(image, Image.Image):
        raise ValueError(f"{op_name}: image 需为 IMAGE 对象，got "
                         f"{type(image).__name__}")
    variant = _parse_key(model, "sam", SAM_VARIANTS, op_name)
    pts, labels = _parse_points(points, op_name)
    bx = _parse_box(box, op_name)
    if not pts and bx is None:
        raise ValueError(f"{op_name}: 至少需要一个提示（points 或 box）")
    mdl = _load_sam(variant, _DTYPES.get(f"sam:{variant}", "fp32"), op_name)
    proc = _sam_proc(f"sam:{variant}", variant, op_name)
    import numpy as np
    import torch

    kw = {"input_points": [pts], "input_labels": [labels]} if pts else {}
    if bx is not None:
        kw["input_boxes"] = [[bx]]
    inputs = proc(image.convert("RGB"), return_tensors="pt", **kw)
    emb, osz, rsz = _embeddings(f"sam:{variant}", mdl, proc, image)
    # 提示张量必须与 image_embeddings 同设备（否则 prompt_encoder 里 CPU×CUDA 报错）
    ip = inputs.get("input_points")
    il = inputs.get("input_labels")
    ib = inputs.get("input_boxes")
    with torch.no_grad():
        out = mdl(image_embeddings=emb,
                  input_points=ip.to("cuda") if ip is not None else None,
                  input_labels=il.to("cuda") if il is not None else None,
                  input_boxes=ib.to("cuda") if ib is not None else None,
                  multimask_output=bool(multimask))
    masks = _norm_masks(proc.image_processor.post_process_masks(
        out.pred_masks.cpu(), osz, rsz), op_name)  # (n, H, W) bool
    scores = out.iou_scores.detach().cpu().numpy().reshape(-1)
    n = int(masks.shape[0])
    idx = max(0, min(n - 1, int(pick)))
    cands = [{"index": i,
              "score": round(float(scores[i]), 4),
              "area": int(masks[i].sum())} for i in range(n)]
    mask_pil = _mask_to_pil(masks[idx])
    ann = _annotated(image, bx, pts, labels, mask_pil, overlay)
    fg = float((np.asarray(mask_pil) > 0).mean())
    print(f"[{op_name}] {variant} {image.size} points={len(pts)} box={bx} "
          f"候选={n} pick={idx} score={cands[idx]['score']} 前景占比={fg:.3f}",
          flush=True)
    return {"mask": mask_pil, "candidates": _dumps(cands),
            "score": float(scores[idx]), "annotated": ann,
            "width": image.size[0], "height": image.size[1]}


def _mask_iou(a, b):
    import numpy as np
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union) if union else 0.0


def _montage(items, cols=None, tile=160, labels=True):
    """蒙版/图片列表 → 带序号的拼接网格图（供 UI 点选）。

    返回 (canvas, cols, tile)：cols/tile 一并回传，供前端把点击位置换算成蒙版下标。
    """
    n = len(items)
    cols = cols or max(1, min(8, int(n ** 0.5) + 1))
    rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile, rows * tile), (24, 24, 26))
    d = ImageDraw.Draw(canvas)
    for i, im in enumerate(items):
        x, y = (i % cols) * tile, (i // cols) * tile
        t = im.convert("RGB")
        t.thumbnail((tile - 8, tile - 8), Image.BILINEAR)
        canvas.paste(t, (x + (tile - t.width) // 2, y + (tile - t.height) // 2))
        if labels:
            pad = max(2, tile // 40)
            d.rectangle([x, y, x + tile - 1, y + tile - 1], outline=(70, 70, 74))
            d.rectangle([x + pad, y + pad, x + pad + 34, y + pad + 16],
                        fill=(0, 0, 0))
            d.text((x + pad + 3, y + pad + 3), str(i), fill=(255, 220, 80))
    return canvas, cols, tile


def sam_auto(image, model="sam:vit_l", points_per_side=16, pred_iou_thresh=0.88,
             stability_score_thresh=0.95, min_area=200, max_masks=64,
             keep="", sort="area", overlay=0.45):
    """SAM1 自动全图分割：IMAGE → 编号网格图 + 选中并集蒙版 + 各蒙版对象 ID。

    - points_per_side：点阵密度（16 → 16×16 个提示；越大越全越慢）
    - min_area：过滤小面积（像素）；max_masks：输出上限
    - keep：'0,2,5' 选择保留的下标（空 = 全选）
    - sort：area/score/index（决定编号顺序，编号即 keep 用的下标）
    """
    op_name = f"seg.sam_auto@{OP_VERSION}"
    if not isinstance(image, Image.Image):
        raise ValueError(f"{op_name}: image 需为 IMAGE 对象，got "
                         f"{type(image).__name__}")
    variant = _parse_key(model, "sam", SAM_VARIANTS, op_name)
    pps = int(points_per_side)
    if pps < 4 or pps > 64:
        raise ValueError(f"{op_name}: points_per_side 需在 4~64，got {pps}")
    mdl = _load_sam(variant, _DTYPES.get(f"sam:{variant}", "fp32"), op_name)
    proc = _sam_proc(f"sam:{variant}", variant, op_name)
    import numpy as np
    import torch
    from transformers import pipeline

    gen = pipeline("mask-generation", model=mdl,
                   image_processor=proc.image_processor, device=0,
                   points_per_batch=64)
    with torch.no_grad():
        res = gen(image.convert("RGB"), points_per_side=pps,
                  pred_iou_thresh=float(pred_iou_thresh),
                  stability_score_thresh=float(stability_score_thresh),
                  crops_n_layers=0)
    raw = res.get("masks")
    raw = list(raw) if raw is not None else []
    sc = res.get("scores")
    scores = [float(s) for s in sc] if sc is not None else [0.0] * len(raw)
    cand = []
    for m, s in zip(raw, scores):
        arr = _as_single(m)
        area = int(arr.sum())
        if area >= int(min_area):
            cand.append((arr, float(s), area))
    # 面积降序 + 掩码 IoU 去重（贪心 NMS，纯 numpy）
    cand.sort(key=lambda t: -t[2])
    kept = []
    for arr, s, area in cand[:256]:
        if all(_mask_iou(arr, k[0]) <= 0.75 for k in kept):
            kept.append((arr, s, area))
        if len(kept) >= 256:
            break
    key = str(sort or "area").strip().lower()
    if key == "score":
        kept.sort(key=lambda t: -t[1])
    elif key != "index":
        kept.sort(key=lambda t: -t[2])
    kept = kept[:max(1, int(max_masks))]

    def _pil(arr):
        return Image.fromarray(arr.astype("uint8") * 255, "L")

    masks = [k[0] for k in kept]
    sel = _parse_keep(keep, len(masks), op_name)
    union = np.zeros(masks[0].shape, dtype=bool) if masks else None
    for i in sel:
        union = np.logical_or(union, masks[i])
    union_pil = _pil(union) if union is not None else _pil(np.zeros(
        (image.size[1], image.size[0]), dtype=bool))
    montage, m_cols, m_tile = _montage([_pil(m) for m in masks])
    ann = _annotated(image, None, None, None, union_pil, overlay)
    # 各蒙版单独进对象仓库（STRING 端口带 ID 列表，供后续节点/UI 取用）
    ids, sizes = [], []
    try:
        from app.main import store
        for arr, s, area in kept:
            oid = store.put("IMAGE", _pil(arr))
            ids.append(oid)
    except Exception as e:  # 无 app 上下文（单测）时跳过
        print(f"[{op_name}] 蒙版入库跳过：{type(e).__name__}: {e}", flush=True)
    ys, xs = np.where(union) if union is not None and union.any() else ((0,), (0,))
    for i, (arr, s, area) in enumerate(kept):
        ysr, xsr = np.where(arr)
        bbox = [int(xsr.min()), int(ysr.min()), int(xsr.max()), int(ysr.max())] \
            if len(xsr) else [0, 0, 0, 0]
        sizes.append({"index": i, "area": area, "score": round(s, 4),
                      "bbox": bbox, "id": ids[i] if i < len(ids) else ""})
    print(f"[{op_name}] {variant} {image.size} pps={pps} 候选={len(raw)} "
          f"去重后={len(kept)} 选中={len(sel)} 并集前景占比="
          f"{float(union.mean()) if union is not None else 0.0:.3f}", flush=True)
    return {"masks": montage, "selected": union_pil, "selected_overlay": ann,
            "mask_ids": _dumps(ids), "count": len(kept), "kept": len(sel),
            "sizes": _dumps(sizes),
            "montage_cols": m_cols, "montage_tile": m_tile,
            "union_area": int(union.sum()) if union is not None else 0,
            "bbox": _dumps([int(xs.min()), int(ys.min()),
                            int(xs.max()), int(ys.max())])}


def _parse_keep(s, n, op_name):
    text = str(s or "").strip()
    if not text:
        return list(range(n))
    out = []
    for chunk in text.replace(" ", ",").replace(";", ",").split(","):
        if not chunk:
            continue
        try:
            i = int(float(chunk))
        except ValueError:
            raise ValueError(f"{op_name}: keep 片段非法 {chunk!r}（形如 '0,2,5'）")
        if i < 0 or i >= n:
            raise ValueError(f"{op_name}: keep 下标 {i} 越界（0~{n - 1}）")
        if i not in out:
            out.append(i)
    return out


def grounding_load(variant="tiny", dtype="fp32"):
    """装载 GroundingDINO → 模型键（STRING）。"""
    op_name = f"seg.grounding_load@{OP_VERSION}"
    variant = str(variant or "tiny").strip().lower()
    path = _variant_dir(GROUNDING_DIR, variant, GROUNDING_VARIANTS, op_name)
    model = _load_grounding(variant, dtype, op_name)
    n = sum(p.numel() for p in model.parameters())
    from app.main import models
    key = f"gd:{variant}"
    vram_mb = int(models._sizes.get(key, 0) / 1024 ** 2)
    print(f"[{op_name}] {variant} params={n / 1e6:.1f}M vram≈{vram_mb}MB "
          f"dtype={_DTYPES.get(key)} 常驻={_residents(op_name)}", flush=True)
    return {"model": key, "params_mb": int(n * 4 / 1024 ** 2), "vram_mb": vram_mb}


def _norm_prompt(text):
    """GroundingDINO 提示词规范化：小写 + 每类以 '.' 结尾（官方格式）。

    HF 官方示例为 `"a cat. a remote control."`；实测裸 `"person"`（无尾点）会 0 命中，
    而 `"person . dress"` 正常 ⇒ 统一按 '.' 切分再逐类补尾点。
    """
    parts = [p.strip() for p in str(text or "").replace(";", ".").split(".")]
    parts = [p for p in parts if p]
    return " ".join(f"{p.lower()}." for p in parts)


def grounding_detect(image, model="gd:tiny", prompt="person", box_threshold=0.35,
                     text_threshold=0.25, max_boxes=10, sam_model="", overlay=0.5):
    """GroundingDINO 文本提示检测 → 框/标签/分数；`sam_model` 非空则框再喂 SAM 出像素级蒙版。

    - prompt：多类用 ' . ' 分隔（如 'person . red dress'），内部统一小写
    - box_threshold/text_threshold：检测置信度门限
    - sam_model：如 'sam:vit_l'（先跑 sam_load）；给出后 mask = 各框的 SAM 蒙版并集
    """
    op_name = f"seg.grounding_detect@{OP_VERSION}"
    if not isinstance(image, Image.Image):
        raise ValueError(f"{op_name}: image 需为 IMAGE 对象，got "
                         f"{type(image).__name__}")
    text = _norm_prompt(prompt)
    if not text:
        raise ValueError(f"{op_name}: prompt 不能为空")
    variant = _parse_key(model, "gd", GROUNDING_VARIANTS, op_name)
    mdl = _load_grounding(variant, _DTYPES.get(f"gd:{variant}", "fp32"), op_name)
    proc = _gd_proc(f"gd:{variant}", variant, op_name)
    import torch

    src = image.convert("RGB")
    inputs = proc(images=src, text=text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = mdl(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, inputs["input_ids"], box_threshold=float(box_threshold),
        text_threshold=float(text_threshold),
        target_sizes=[src.size[::-1]])[0]
    scores = res["scores"].detach().cpu().tolist()
    boxes = [[float(v) for v in b] for b in res["boxes"].detach().cpu().tolist()]
    labels = res.get("labels")
    if labels is None:
        labels = res.get("text_labels")
    labels = list(labels) if labels is not None else [""] * len(boxes)
    order = sorted(range(len(scores)), key=lambda i: -scores[i])[:max(1, int(max_boxes))]
    boxes = [boxes[i] for i in order]
    labels = [labels[i] for i in order]
    scores = [scores[i] for i in order]

    # mask：sam_model 非空 → 像素级；否则框矩形并集
    import numpy as np
    from PIL import ImageDraw as _ID
    union = np.zeros((src.size[1], src.size[0]), dtype=bool)
    used_sam = False
    if str(sam_model or "").strip() and boxes:
        for b in boxes:
            m = sam(src, model=str(sam_model).strip(),
                    box=",".join(str(round(v, 2)) for v in b), multimask=False,
                    overlay=0.0)["mask"]
            union = np.logical_or(union, np.asarray(m) > 0)
        used_sam = True
    else:
        tmp = Image.new("L", src.size, 0)
        d = _ID.Draw(tmp)
        for b in boxes:
            d.rectangle(b, fill=255)
        union = np.asarray(tmp) > 0
    mask_pil = Image.fromarray(union.astype("uint8") * 255, "L")

    ann = _annotated(src, None, None, None, mask_pil if used_sam else None, overlay)
    d = ImageDraw.Draw(ann)
    for b, lab, sc in zip(boxes, labels, scores):
        d.rectangle(b, outline=(255, 160, 0), width=3)
        d.text((b[0] + 4, max(0, b[1] - 14)), f"{lab} {sc:.2f}", fill=(255, 230, 0))
    print(f"[{op_name}] {variant} {src.size} prompt={text!r} 命中={len(boxes)} "
          f"sam_refine={used_sam} labels={labels} "
          f"scores={[round(s, 3) for s in scores]}", flush=True)
    return {"annotated": ann, "boxes": _dumps(boxes),
            "labels": _dumps(labels),
            "scores": _dumps([round(s, 4) for s in scores]),
            "count": len(boxes), "mask": mask_pil,
            "width": src.size[0], "height": src.size[1]}


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "seg.sam_load",
        inputs={"variant": "STRING", "dtype": "STRING"},
        outputs={"model": "STRING", "params_mb": "INT", "vram_mb": "INT"},
        description="装载 SAM1 变体（variant=vit_h|vit_l|vit_b；权重在 "
                    "SAM_MODELS_DIR/<variant>/）→ 模型键 'sam:<variant>'，"
                    "注册进 model_manager LRU（可 model-unload / LRU 淘汰）。"
                    "dtype 默认 fp32（Pascal fp16 吞吐 1/64）"
    )(sam_load)
    registry.register(
        "seg.sam",
        inputs={"image": "IMAGE", "model": "STRING", "points": "STRING",
                "box": "STRING", "multimask": "BOOL", "pick": "INT",
                "overlay": "FLOAT"},
        outputs={"mask": "IMAGE", "candidates": "STRING", "score": "FLOAT",
                 "annotated": "IMAGE", "width": "INT", "height": "INT"},
        description="SAM1 点/框提示分割：points='x,y,label;…'（1=正 0=负）、"
                    "box='x1,y1,x2,y2'（可同时给）→ 灰度蒙版 + 多候选分数 + "
                    "叠加预览。image embedding 按图片内容缓存 ⇒ 连续点选不重复编码"
    )(sam)
    registry.register(
        "seg.sam_auto",
        inputs={"image": "IMAGE", "model": "STRING", "points_per_side": "INT",
                "pred_iou_thresh": "FLOAT", "stability_score_thresh": "FLOAT",
                "min_area": "INT", "max_masks": "INT", "keep": "STRING",
                "sort": "STRING", "overlay": "FLOAT"},
        outputs={"masks": "IMAGE", "selected": "IMAGE",
                 "selected_overlay": "IMAGE", "mask_ids": "STRING",
                 "count": "INT", "kept": "INT", "sizes": "STRING",
                 "montage_cols": "INT", "montage_tile": "INT",
                 "union_area": "INT", "bbox": "STRING"},
        description="SAM1 自动全图分割（点阵提示 + IoU 去重）→ 编号网格图 + "
                    "keep 下标并集蒙版 + 各蒙版对象 ID（mask_ids）。"
                    "montage_cols/montage_tile 回传网格布局供前端点选换算。keep 空=全选"
    )(sam_auto)
    registry.register(
        "seg.grounding_load",
        inputs={"variant": "STRING", "dtype": "STRING"},
        outputs={"model": "STRING", "params_mb": "INT", "vram_mb": "INT"},
        description="装载 GroundingDINO（variant=tiny|base；权重在 "
                    "GROUNDING_MODELS_DIR/<variant>/）→ 模型键 'gd:<variant>'，"
                    "注册进 model_manager LRU"
    )(grounding_load)
    registry.register(
        "seg.grounding_detect",
        inputs={"image": "IMAGE", "model": "STRING", "prompt": "STRING",
                "box_threshold": "FLOAT", "text_threshold": "FLOAT",
                "max_boxes": "INT", "sam_model": "STRING", "overlay": "FLOAT"},
        outputs={"annotated": "IMAGE", "boxes": "STRING", "labels": "STRING",
                 "scores": "STRING", "count": "INT", "mask": "IMAGE",
                 "width": "INT", "height": "INT"},
        description="文本提示检测（GroundingDINO）：prompt='person . red dress' → "
                    "框/标签/分数 + 叠加图；sam_model 非空则把框再喂 SAM 得"
                    "像素级蒙版（文本→精确蒙版一条链）"
    )(grounding_detect)
