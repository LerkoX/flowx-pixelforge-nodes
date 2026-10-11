"""分块重绘超分算子族（G4 / B3 批次，dev-plan §四十九）。

算子：
- `tiled.plan`  ：**纯计算 + PIL 画网格，零 GPU** ⇒ 切片网格计划（JSON）+ 网格覆盖层预览
- `tiled.apply` ：按计划逐块 img2img 重绘 + **权重归一化无缝融合** → 超分图

为什么在服务端一次做完（而不是节点侧循环）：
  逐块走网络 = N 次往返 + 每块重新准备条件/管道，且中途无法保证显存租约；本族用服务端
  核心原语 `app.ops.resolve_pipe / vae_encode / sample / vae_decode`（`detail-refine` 同款）
  ⇒ **服务端 app/ 零改动**，一块 job 内跑完全部 tile，并借 `app.jobs.set_progress` 上报逐块进度。

关键口径（dev-plan §49.4）：
- 网格：`tiled.plan` 与 `tiled.apply` **共用 `_grid()` 单一事实源**（plan 与实际执行永不脱节）；
  每块坐标对齐 **8 的倍数**（VAE 下采样要求，`detail-refine` 同款纪律）。
- 确定性：每块 seed = `seed + index`，固定块序 ⇒ 同参数逐字节可复现。
- **无缝融合**：不用 paste，而是权重归一化累加 `out = Σ(tile·W) / ΣW`，W 在 overlap 区线性
  降到 0 ⇒ overlap 区两侧权重和为 1，**数学上无偏、无权重断层**；`seam_fix=mask` 再把
  `ΣW` 低值带二次低 denoise 重绘后重新融合（Ultimate SD Upscale 的 seam fix 精神）。
- 预放大：`pre=lanczos`（PIL）/ `pre=model`（复用 `image.upscale_with_model`）/ `pre=none`。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "91cb9a1d3d9a"
OP_FAMILY = "tiled"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


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

from PIL import Image, ImageDraw, ImageFilter

ALIGN = 8          # 块坐标/尺寸对齐（VAE 下采样）
MIN_TILE = 128
MAX_TILES = 4096   # 防御性上限（块数爆炸时早失败）


# ---------------------------------------------------------------------------
# 网格（单一事实源）
# ---------------------------------------------------------------------------

def _dumps(obj):
    """JSON 负载的信封编码：`@` + JSON（见 seg.py 同名函数说明：防 Studio 自动反序列化）。"""
    return "@" + json.dumps(obj, ensure_ascii=False)


def _loads(text, default=None):
    s = str(text or "").strip()
    if s.startswith("@"):
        s = s[1:]
    if not s:
        return default
    return json.loads(s)


def _split(total, tile, overlap):
    """一维切分：返回 [(start, end), ...]，末块贴边（保证覆盖到 total）。"""
    step = max(1, int(tile) - int(overlap))
    spans = []
    pos = 0
    while True:
        end = min(total, pos + int(tile))
        spans.append((pos, end))
        if end >= total:
            break
        pos += step
        if pos >= total:
            break
    return spans


def _target_size(w, h, scale, target_w, target_h, op_name):
    tw = int(target_w or 0)
    th = int(target_h or 0)
    if tw <= 0 or th <= 0:
        s = float(scale or 0)
        if s <= 0:
            raise ValueError(f"{op_name}: scale 需 > 0（或显式给 target_w/target_h）")
        tw, th = round(w * s), round(h * s)
    else:
        s = tw / float(w)
    tw = max(64, tw // ALIGN * ALIGN)
    th = max(64, th // ALIGN * ALIGN)
    return tw, th, s


def _grid(w, h, tw, th, tile, overlap, op_name):
    """切片计划：目标尺寸 (tw,th) 上按 (tile, overlap) 切网格（坐标对齐 ALIGN）。"""
    tile = int(tile)
    overlap = int(overlap)
    if tile < MIN_TILE:
        raise ValueError(f"{op_name}: tile 需 ≥ {MIN_TILE}，got {tile}")
    if overlap < 0 or overlap >= tile // 2:
        raise ValueError(f"{op_name}: overlap 需在 0 ~ tile/2，got {overlap}")
    xs = _split(tw, tile, overlap)
    ys = _split(th, tile, overlap)
    cells = []
    for yi, (y0, y1) in enumerate(ys):
        for xi, (x0, x1) in enumerate(xs):
            # 坐标/尺寸对齐（首块不动，末块贴边）
            x0a, y0a = (x0 // ALIGN) * ALIGN, (y0 // ALIGN) * ALIGN
            x1a, y1a = x0a + (x1 - x0), y0a + (y1 - y0)
            x1a, y1a = min(tw, x1a), min(th, y1a)
            cells.append({"index": len(cells), "x": x0a, "y": y0a,
                          "w": max(ALIGN, x1a - x0a),
                          "h": max(ALIGN, y1a - y0a),
                          "row": yi, "col": xi})
    if len(cells) > MAX_TILES:
        raise ValueError(f"{op_name}: 块数 {len(cells)} 超过上限 {MAX_TILES}"
                         f"（调大 tile 或调小 overlap/scale）")
    return cells


def _parse_plan(plan, op_name):
    """执行侧解析 plan JSON（缺字段早失败，给出人话原因）。"""
    try:
        d = _loads(plan)
    except Exception as e:
        raise ValueError(f"{op_name}: plan 不是合法 JSON（{e}）；留空则按本节点参数自算")
    need = ("target", "tile", "overlap", "cells", "seed", "seam_fix")
    missing = [k for k in need if k not in d]
    if missing:
        raise ValueError(f"{op_name}: plan 缺字段 {missing}"
                         f"（应由 tiled-upscale-plan 节点产出）")
    return d


def _weights(w, h, overlap, feather=True):
    """单块羽化权重图（L 模式）：内部 255，四边 overlap 宽内线性降到 ~0。

    用 distance transform 的近似写法（numpy 逐行/逐列渐变）避免依赖 scipy：
    权重 = min(top, bottom, left, right 四个方向的渐变值)。
    """
    import numpy as np
    o = max(1, int(overlap))
    ramp_x = np.ones(w, dtype=np.float32)
    for i in range(min(o, w)):
        v = (i + 0.5) / o
        ramp_x[i] = min(ramp_x[i], v)
        ramp_x[w - 1 - i] = min(ramp_x[w - 1 - i], v)
    ramp_y = np.ones(h, dtype=np.float32)
    for j in range(min(o, h)):
        v = (j + 0.5) / o
        ramp_y[j] = min(ramp_y[j], v)
        ramp_y[h - 1 - j] = min(ramp_y[h - 1 - j], v)
    W = np.minimum(ramp_y[:, None], ramp_x[None, :])
    return Image.fromarray((W * 255.0).round().astype("uint8"), "L")


def _overlay_grid(image, cells, tile, overlap):
    """网格覆盖层预览：块边框 + 序号 + overlap 阴影。"""
    base = image.convert("RGB").copy()
    d = ImageDraw.Draw(base, "RGBA")
    for c in cells:
        d.rectangle([c["x"], c["y"], c["x"] + c["w"] - 1, c["y"] + c["h"] - 1],
                    outline=(255, 90, 90, 230), width=2)
        if overlap > 0:
            d.rectangle([c["x"], c["y"], c["x"] + c["w"] - 1, c["y"] + c["h"] - 1],
                        outline=(80, 200, 255, 120), width=1)
        d.rectangle([c["x"] + 3, c["y"] + 3, c["x"] + 40, c["y"] + 20],
                    fill=(0, 0, 0, 170))
        d.text((c["x"] + 6, c["y"] + 6), str(c["index"]), fill=(255, 230, 90, 255))
    return base


def _apply_upscale_model(image, handle, op_name):
    """调用已注册的 `image.upscale_with_model` 算子（插件算子间互调）。

    插件算子之间没有直接的函数引用，但注册表里有：按前缀找到实际注册名
    （可能带 F1 版本后缀），再调 `registry.get(name).fn`。"""
    from app.main import registry
    target = None
    for spec in registry.list():
        n = spec.get("name", "")
        if n == "image.upscale_with_model" or n.startswith("image.upscale_with_model@"):
            target = n
            break
    if target is None:
        raise RuntimeError(
            f"{op_name}: pre=model 需要算子 image.upscale_with_model"
            f"（随 upscale-model-apply 节点插件注册；也可改为上游堆叠该节点）")
    out = registry.get(target).fn(image, handle, 0)
    return out["image"] if isinstance(out, dict) else out


def _pre_scale(image, tw, th, pre, upscale_model=None, op_name=""):
    """预放大到目标尺寸（pre=lanczos 走 PIL，model 走 spandrel 算子放大，none 不放大）。"""
    mode = str(pre or "lanczos").strip().lower()
    if mode == "model":
        if upscale_model is None:
            raise ValueError(f"{op_name}: pre=model 需要 upscale_model 输入"
                             f"（upscale-model-load 的输出）")
        img = _apply_upscale_model(image.convert("RGB"), upscale_model, op_name)
        if img.size != (tw, th):
            img = img.resize((tw, th), Image.LANCZOS)
        return img
    if mode == "none":
        return image.convert("RGB")
    if mode != "lanczos":
        raise ValueError(f"{op_name}: pre 非法 {pre!r}，可选 none/lanczos/model")
    return image.convert("RGB").resize((tw, th), Image.LANCZOS)


def _merge(acc, wsum, tile_img, weight, box):
    """权重归一化累加（acc/wsum 均为 float32 numpy，尺寸 = 目标尺寸）。"""
    import numpy as np
    x, y, w, h = box
    t = np.asarray(tile_img.convert("RGB"), dtype=np.float32)
    W = np.asarray(weight, dtype=np.float32)[:, :, None] / 255.0
    if t.shape[0] != h or t.shape[1] != w:  # 采样器可能回不同尺寸：强制对齐
        t = np.asarray(Image.fromarray(t.astype("uint8")).resize((w, h),
                                                                 Image.LANCZOS),
                       dtype=np.float32)
    acc[y:y + h, x:x + w] += t * W
    wsum[y:y + h, x:x + w] += W[:, :, 0]


# ---------------------------------------------------------------------------
# 算子
# ---------------------------------------------------------------------------

def tiled_plan(image, scale=2.0, target_w=0, target_h=0, tile=768, overlap=96,
               pre="lanczos", denoise=0.35, seed=-1, seam_fix="mask",
               band_denoise=0.25, steps=20):
    """分块重绘超分：切片网格计划（纯计算，零 GPU）→ plan JSON + 网格覆盖层。

    返回的 `plan` 直接喂 `tiled.apply` 的 plan 参数 ⇒ 预览与实际执行**同源**。
    """
    op_name = f"tiled.plan@{OP_VERSION}"
    if not isinstance(image, Image.Image):
        raise ValueError(f"{op_name}: image 需为 IMAGE 对象，got "
                         f"{type(image).__name__}")
    w, h = image.size
    tw, th, sc = _target_size(w, h, scale, target_w, target_h, op_name)
    cells = _grid(w, h, tw, th, tile, overlap, op_name)
    sf = str(seam_fix or "mask").strip().lower()
    if sf not in ("mask", "band", "off"):
        raise ValueError(f"{op_name}: seam_fix 非法 {seam_fix!r}，可选 mask/band/off")
    pm = str(pre or "lanczos").strip().lower()
    if pm not in ("none", "lanczos", "model"):
        raise ValueError(f"{op_name}: pre 非法 {pre!r}，可选 none/lanczos/model")
    s = int(seed)
    plan = {"src": [w, h], "target": [tw, th], "scale": round(sc, 4),
            "tile": int(tile), "overlap": int(overlap), "pre": pm,
            "denoise": float(denoise), "band_denoise": float(band_denoise),
            "steps": int(steps), "seed": s, "seam_fix": sf, "cells": cells}
    est = len(cells) * int(steps) * (2 if sf in ("mask", "band") else 1)
    overlay = _overlay_grid(image.resize((tw, th), Image.LANCZOS), cells,
                            int(tile), int(overlap))
    print(f"[{op_name}] {w}x{h} → {tw}x{th} tile={tile} overlap={overlap} "
          f"块数={len(cells)}（{len(_split(th, int(tile), int(overlap)))}行 × "
          f"{len(_split(tw, int(tile), int(overlap)))}列）估采样步={est} "
          f"seam_fix={sf}", flush=True)
    return {"plan": _dumps(plan), "grid_image": overlay,
            "tiles": len(cells), "rows": len(_split(th, int(tile), int(overlap))),
            "cols": len(_split(tw, int(tile), int(overlap))),
            "target_w": tw, "target_h": th, "est_samples": est,
            "target_size": f"{tw}x{th}"}


def tiled_apply(image, model, pos, neg, plan="", scale=2.0, target_w=0,
                target_h=0, tile=768, overlap=96, pre="lanczos", denoise=0.35,
                seed=-1, seam_fix="mask", band_denoise=0.25, steps=20, cfg=7.0,
                sampler_name="", scheduler="normal", upscale_model=None):
    """按切片计划逐块 img2img 重绘 + 权重归一化无缝融合 → 超分图。

    `plan` 非空则用 `tiled-upscale-plan` 的口径（推荐）；空则用本节点参数自算（同源同算法）。
    """
    op_name = f"tiled.apply@{OP_VERSION}"
    if not isinstance(image, Image.Image):
        raise ValueError(f"{op_name}: image 需为 IMAGE 对象，got "
                         f"{type(image).__name__}")
    if plan and str(plan).strip():
        d = _parse_plan(plan, op_name)
        tw, th = int(d["target"][0]), int(d["target"][1])
        tile, overlap = int(d["tile"]), int(d["overlap"])
        pre, denoise = d["pre"], float(d["denoise"])
        band_denoise = float(d["band_denoise"])
        seed, seam_fix = int(d["seed"]), d["seam_fix"]
        steps = int(d["steps"])
        cells = d["cells"]
    else:
        w0, h0 = image.size
        tw, th, _ = _target_size(w0, h0, scale, target_w, target_h, op_name)
        cells = _grid(w0, h0, tw, th, tile, overlap, op_name)
        seam_fix = str(seam_fix or "mask").strip().lower()
    if seam_fix not in ("mask", "band", "off"):
        raise ValueError(f"{op_name}: seam_fix 非法 {seam_fix!r}，可选 mask/band/off")
    steps = max(1, int(steps))
    cfg = float(cfg)

    import numpy as np
    import torch
    from app import ops

    pipe, _ = ops.resolve_pipe(model)
    sampler_name = str(sampler_name or "").strip() or ops.DEFAULT_SAMPLER
    s = int(seed)
    # 基种子：>=0 用给定值；-1 抽一个并落进日志（保证同一次执行内各块可复现）
    base = s if s >= 0 else int(torch.randint(0, 2 ** 31 - 1, (1,)).item())
    print(f"[{op_name}] 基种子={base}（每块 seed=基种子+块序号） sampler={sampler_name} "
          f"scheduler={scheduler}", flush=True)
    big = _pre_scale(image, tw, th, pre, upscale_model, op_name)
    pre_snapshot = big.copy()
    acc = np.zeros((th, tw, 3), dtype=np.float32)
    wsum = np.zeros((th, tw), dtype=np.float32)
    weights = {}

    prog = _progressor(op_name)
    n = len(cells)
    for i, c in enumerate(cells):
        box = (c["x"], c["y"], c["w"], c["h"])
        t = big.crop((c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]))
        lat = ops.vae_encode(pipe, t)["latent"]
        out = ops.sample(model, pos, neg, lat, seed=base + c["index"], steps=steps,
                         cfg=cfg, sampler_name=sampler_name, scheduler=scheduler,
                         denoise=float(denoise))
        redraw = ops.vae_decode(pipe, out["latent"])["image"]
        if not isinstance(redraw, Image.Image):
            redraw = redraw[0]
        W = weights.setdefault((c["w"], c["h"]), _weights(c["w"], c["h"], overlap))
        _merge(acc, wsum, redraw, W, box)
        prog(i + 1, n, f"tile {i + 1}/{n}")

    if seam_fix == "band":
        # 只对 ΣW 低值（接缝带）做二次低 denoise 重绘
        band = np.asarray(_band_mask(wsum, overlap)) > 0
        if band.any():
            idx = np.argwhere(band)
            y0, y1 = int(idx[:, 0].min()), int(idx[:, 0].max()) + 1
            x0, x1 = int(idx[:, 1].min()), int(idx[:, 1].max()) + 1
            strip = _from_acc(acc, wsum).crop((x0, y0, x1, y1))
            lat = ops.vae_encode(pipe, strip)["latent"]
            out = ops.sample(model, pos, neg, lat, seed=base + 70000,
                             steps=steps, cfg=cfg, sampler_name=sampler_name,
                             scheduler=scheduler, denoise=float(band_denoise))
            fixed = ops.vae_decode(pipe, out["latent"])["image"]
            if not isinstance(fixed, Image.Image):
                fixed = fixed[0]
            full = _from_acc(acc, wsum)
            full.paste(fixed.resize((x1 - x0, y1 - y0), Image.LANCZOS), (x0, y0),
                       Image.fromarray(band[y0:y1, x0:x1].astype("uint8") * 255, "L"))
            acc = np.asarray(full, dtype=np.float32)
            wsum = np.ones_like(wsum)
        print(f"[{op_name}] seam_fix=band 接缝带二次重绘完成", flush=True)

    final = _from_acc(acc, wsum)
    seam_map = Image.fromarray((np.clip(wsum, 0, 1) * 255).astype("uint8"), "L")
    covered = float((wsum > 0).mean())
    print(f"[{op_name}] {image.size} → {final.size} 块={n} denoise={denoise} "
          f"seam_fix={seam_fix} 覆盖率={covered:.4f} 权重最小值={float(wsum.min()):.3f} "
          f"最大={float(wsum.max()):.3f}", flush=True)
    return {"image": final, "pre_upscaled": pre_snapshot, "seam_map": seam_map,
            "tiles": n, "coverage": covered, "width": final.size[0],
            "height": final.size[1]}


def _from_acc(acc, wsum):
    """归一化 + 取整回 PIL。"""
    import numpy as np
    out = acc / np.maximum(wsum, 1e-6)[:, :, None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype("uint8"), "RGB")


def _band_mask(wsum, overlap):
    """接缝带 = ΣW 明显低于 1 的区域（overlap 区被两侧摊薄）。"""
    import numpy as np
    m = ((wsum < 0.92) * 255).astype("uint8")
    # 稍微膨胀，保证把缝包住
    return Image.fromarray(m, "L").filter(
        ImageFilter.MaxFilter(max(3, int(overlap) // 2 * 2 + 1)))


def _progressor(op_name):
    """逐块进度上报（`app.execution.current()` 拿到当前 job；无 job 时退回 print）。"""
    try:
        from app import execution
        job = execution.current()
    except Exception:
        job = None
    last = {"i": 0}

    def report(i, total, msg):
        if job is not None:
            try:
                job.set_progress(i, total)
            except Exception:
                pass
        if i == total or i - last["i"] >= max(1, total // 10):
            last["i"] = i
            print(f"[{op_name}] {msg}", flush=True)

    return report


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "tiled.plan",
        inputs={"image": "IMAGE", "scale": "FLOAT", "target_w": "INT",
                "target_h": "INT", "tile": "INT", "overlap": "INT",
                "pre": "STRING", "denoise": "FLOAT", "seed": "INT",
                "seam_fix": "STRING", "band_denoise": "FLOAT", "steps": "INT"},
        outputs={"plan": "STRING", "grid_image": "IMAGE", "tiles": "INT",
                 "rows": "INT", "cols": "INT", "target_w": "INT",
                 "target_h": "INT", "est_samples": "INT", "target_size": "STRING"},
        description="分块重绘超分·切片规划（纯计算，零 GPU）：目标尺寸 + tile/overlap "
                    "→ 网格计划 JSON（喂 tiled.apply，保证预览与执行同源）+ 网格覆盖层预览"
    )(tiled_plan)
    registry.register(
        "tiled.apply",
        inputs={"image": "IMAGE", "model": "MODEL", "pos": "COND", "neg": "COND",
                "plan": "STRING", "scale": "FLOAT", "target_w": "INT",
                "target_h": "INT", "tile": "INT", "overlap": "INT",
                "pre": "STRING", "denoise": "FLOAT", "seed": "INT",
                "seam_fix": "STRING", "band_denoise": "FLOAT", "steps": "INT",
                "cfg": "FLOAT", "sampler_name": "STRING", "scheduler": "STRING",
                "upscale_model": "UPSCALE_MODEL"},
        outputs={"image": "IMAGE", "pre_upscaled": "IMAGE", "seam_map": "IMAGE",
                 "tiles": "INT", "coverage": "FLOAT", "width": "INT",
                 "height": "INT"},
        description="分块重绘超分：按计划逐块 img2img(denoise) 重绘 + **权重归一化无缝融合**"
                    "（out=Σtile·W/ΣW）→ 超分图 + 预放大图 + 接缝权重图（复核用）。"
                    "整条在一块 job 内跑完并上报逐块进度"
    )(tiled_apply)
