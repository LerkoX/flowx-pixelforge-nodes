"""flux-mask-refine 服务端算子注册文件：注册 flux.mask_refine

瓶颈定位打点（dev-log §62，P1）：环境变量 **FLOWX_PROFILE**（在**推理服务进程**上
设置，默认 0 = 关闭、零开销、行为与旧版逐字节一致）：
  1 = 轻量分段计时：CUDA event 总时长 + 每步墙钟 + 反量化入口累计
      （候选逐个 patch）+ transformer.forward 累计；
  2 = 在 1 之上再挂 torch.profiler，打印 top-20 算子表。
报告经 job.progress["profile"] 旁路回传（GET /jobs/{id} 可取，节点侧再打印），
同时写服务端日志。判据：反量化占比高 ⇒ 坐实 Q4 GGUF 逐层反量化为主因。
"""
import os
import random
import time

import torch

from app import execution, preview

FLUX_DIR = os.environ.get("FLUX_DIR", "/models/flux")


def _profile_level():
    """FLOWX_PROFILE 开关：0=关（默认）/ 1=轻量分段 / 2=+torch.profiler。"""
    try:
        return int(os.environ.get("FLOWX_PROFILE", "0") or 0)
    except (TypeError, ValueError):
        return 0


class _Profiler:
    """单次执行的计时打点（仅诊断用）。反量化入口用嵌套深度去重：
    内外两层都 patch 时只计最外层，避免双计（不同 diffusers/gguf 版本入口位置不同）。"""

    def __init__(self, level, steps, guide_size):
        self.level = level
        self.steps = steps
        self.guide_size = guide_size
        self.dequant = {"n": 0, "t": 0.0}
        self.forward = {"n": 0, "t": 0.0}
        self.cb = 0.0
        self.walls = []
        self._patched = []
        self._depth = 0
        self._t_last = None
        self._prof = None
        self._table = None

    def _wrap(self, holder, attr, stat):
        orig = getattr(holder, attr, None)
        if orig is None or not callable(orig):
            return False
        prof = self

        def wrapper(*a, **kw):
            if prof._depth > 0:      # 嵌套调用：外层已统计，跳过
                return orig(*a, **kw)
            prof._depth += 1
            t = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                prof._depth -= 1
                stat["n"] += 1
                stat["t"] += time.perf_counter() - t

        try:
            setattr(holder, attr, wrapper)
        except Exception:
            return False
        self._patched.append((holder, attr, orig))
        return True

    def patch(self, model):
        # 反量化入口：gguf 库最内层优先，回退 diffusers 的 GGUFParameter.dequantize
        done = False
        try:
            import gguf.quants as _q
            done = self._wrap(_q, "dequantize", self.dequant)
        except Exception:
            done = False
        if not done:
            try:
                import diffusers.quantizers.gguf.utils as _gu
                cls = getattr(_gu, "GGUFParameter", None)
                if cls is not None:
                    self._wrap(cls, "dequantize", self.dequant)
            except Exception:
                pass
        # transformer 整次 forward（采样时每步调用一次）
        self._wrap(type(model), "forward", self.forward)

    def unpatch(self):
        for holder, attr, orig in self._patched:
            try:
                setattr(holder, attr, orig)
            except Exception:
                pass
        self._patched = []

    def start(self):
        self._t_last = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self.walls.append(now - self._t_last if self._t_last is not None else 0.0)
        self._t_last = now

    def profiler(self):
        if self.level < 2:
            return None
        from torch.profiler import ProfilerActivity, profile
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        return self._prof

    def close_profiler(self):
        if self._prof is None:
            return
        self._prof.__exit__(None, None, None)
        try:
            self._table = self._prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=20)
        except Exception as e:
            self._table = f"(profiler table failed: {e})"

    def report(self, total_s, crop):
        def pct(x):
            return f"{x / total_s * 100:.1f}%" if total_s else "n/a"
        out = [f"=== flux.mask_refine PROFILE (level={self.level}) ===",
               f"steps={self.steps} guide_size={self.guide_size} crop={crop}",
               f"TOTAL(cuda event) {total_s:.1f}s | "
               f"dequant {self.dequant['t']:.1f}s ({pct(self.dequant['t'])}, "
               f"{self.dequant['n']} calls) | "
               f"forward {self.forward['t']:.1f}s ({pct(self.forward['t'])}, "
               f"{self.forward['n']} calls) | callback {self.cb:.2f}s"]
        for i, w in enumerate(self.walls, 1):
            out.append(f"  step {i}: {w:.1f}s")
        if self._table:
            out.append(self._table)
        return "\n".join(out)


def _fresh_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(FLUX_DIR, "scheduler"))


def _small(img, max_size=512):
    from PIL import Image
    if max(img.size) <= max_size:
        return img
    scale = max_size / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)),
                      Image.Resampling.LANCZOS)


def register(registry):
    @registry.register(
        "flux.mask_refine",
        inputs={
            "model": "MODEL",
            "vae": "VAE",
            "cond": "COND",
            "image": "IMAGE",
            "mask": "IMAGE",
            "padding": "FLOAT",
            "denoise": "FLOAT",
            "steps": "INT",
            "guidance": "FLOAT",
            "seed": "INT",
            "guide_size": "INT",
            "feather": "INT"
        },
        outputs={"image": "IMAGE"},
        description="FLUX 蒙版局部重绘：根据输入的蒙版外接框外扩裁剪 → 放大到 guide_size → "
                    "FLUX img2img 重绘 → 结合蒙版与边缘羽化无缝贴回原图。"
    )
    def _op_flux_mask_refine(model, vae, cond, image, mask, padding=0.3,
                             denoise=0.8, steps=8, guidance=0.0, seed=-1,
                             guide_size=768, feather=16):
        from diffusers import FluxImg2ImgPipeline
        from PIL import Image, ImageDraw, ImageFilter
        import numpy as np

        if type(model).__name__ != "FluxTransformer2DModel":
            raise ValueError("flux.mask_refine: model 需为 flux.unet_load 输出")
        if not isinstance(cond, dict) or "prompt_embeds" not in cond:
            raise ValueError("flux.mask_refine: cond 需为 flux.encode 输出")
        if not torch.cuda.is_available():
            raise ValueError("flux.mask_refine 需要 CUDA")

        seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**32 - 1)
        steps = int(steps)
        job = execution.current()
        rec = preview.recorder_for(job.id, 1) if job else None

        # 确保 mask 为灰度模式（L）且尺寸与原图一致
        mask_l = mask.convert("L")
        if mask_l.size != image.size:
            mask_l = mask_l.resize(image.size, Image.Resampling.LANCZOS)

        # 寻找蒙版外接框 (bbox)
        mask_binary = mask_l.point(lambda p: 255 if p > 128 else 0)
        bbox = mask_binary.getbbox()

        if bbox is None:
            print("[flux.mask_refine] empty mask, no area to refine", flush=True)
            if job:
                job.set_progress(1, 1)
            return {"image": image}

        W, H = image.size
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * float(padding), bh * float(padding)

        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))

        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            print(f"[flux.mask_refine] crop region too small ({cx2-cx1}x{cy2-cy1}), return orig",
                  flush=True)
            if job:
                job.set_progress(1, 1)
            return {"image": image}

        crop_img = image.crop((cx1, cy1, cx2, cy2))
        crop_mask = mask_l.crop((cx1, cy1, cx2, cy2))

        scale = float(guide_size) / min(crop_img.size)
        tw = max(64, round(crop_img.width * scale) // 16 * 16)
        th = max(64, round(crop_img.height * scale) // 16 * 16)
        crop_img_big = crop_img.resize((tw, th), Image.Resampling.LANCZOS)

        if rec:
            dbg = image.copy()
            d = ImageDraw.Draw(dbg)
            d.rectangle([cx1, cy1, cx2, cy2], outline=(34, 211, 238), width=3)
            rec.push_pil(_small(dbg), 0.0)

        t0 = time.time()

        level = _profile_level()
        prof = _Profiler(level, steps, guide_size) if level >= 1 else None
        if prof:
            print(f"[flux.mask_refine] PROFILE level={level} ON "
                  f"(steps={steps} guide_size={guide_size})", flush=True)
            prof.patch(model)

        model.to("cuda")
        vae.to("cuda")
        torch.cuda.empty_cache()

        i2i = FluxImg2ImgPipeline(
            scheduler=_fresh_scheduler(), vae=vae, text_encoder=None,
            tokenizer=None, text_encoder_2=None, tokenizer_2=None,
            transformer=model)
        i2i.set_progress_bar_config(disable=True)

        pe = cond["prompt_embeds"].to("cuda", model.dtype)
        pooled = cond["pooled"].to("cuda", model.dtype)

        def _cb(p, si, t, kw):
            execution.check_cancelled()
            if job:
                job.set_progress(si + 1, steps)
            if prof:
                prof.tick()
            return kw

        i2i_prof = prof.profiler() if prof else None
        if i2i_prof:
            i2i_prof.__enter__()
        ev0 = torch.cuda.Event(enable_timing=True) if prof else None
        if ev0:
            ev0.record()
        if prof:
            prof.start()
        refined_big = i2i(
            prompt_embeds=pe, pooled_prompt_embeds=pooled,
            image=crop_img_big, strength=float(denoise),
            num_inference_steps=steps, guidance_scale=float(guidance),
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=_cb,
            callback_on_step_end_tensor_inputs=[],
        ).images[0]
        gpu_total = 0.0
        if ev0:
            ev1 = torch.cuda.Event(enable_timing=True)
            ev1.record()
            torch.cuda.synchronize()
            gpu_total = ev0.elapsed_time(ev1) / 1000.0
        if prof:
            prof.close_profiler()

        refined = refined_big.resize(crop_img.size, Image.Resampling.LANCZOS)

        feather_val = int(feather)
        border_mask = Image.new("L", crop_img.size, 0)
        if feather_val > 0 and crop_img.width > 2 * feather_val and crop_img.height > 2 * feather_val:
            ImageDraw.Draw(border_mask).rectangle(
                [feather_val, feather_val, crop_img.width - feather_val, crop_img.height - feather_val],
                fill=255
            )
            border_mask = border_mask.filter(ImageFilter.GaussianBlur(feather_val / 2))
        else:
            border_mask = Image.new("L", crop_img.size, 255)

        m1 = np.array(crop_mask, dtype=np.float32) / 255.0
        m2 = np.array(border_mask, dtype=np.float32) / 255.0
        paste_m = np.clip(m1 * m2 * 255.0, 0, 255).astype(np.uint8)
        paste_mask = Image.fromarray(paste_m, mode="L")

        out_img = image.copy()
        out_img.paste(refined, (cx1, cy1), paste_mask)

        if job:
            job.set_progress(1, 1)
        if rec:
            rec.push_pil(_small(out_img), 1.0)

        print(f"[flux.mask_refine] crop ({cx1},{cy1},{cx2},{cy2}) -> {tw}x{th} "
              f"denoise={denoise} steps={steps} ({time.time()-t0:.1f}s)",
              flush=True)

        if prof:
            rep = prof.report(gpu_total, f"{tw}x{th}")
            print(rep, flush=True)
            prof.unpatch()
            if job:                 # 最后塞入（set_progress 会重置 progress dict）
                job.progress["profile"] = rep

        return {"image": out_img}
