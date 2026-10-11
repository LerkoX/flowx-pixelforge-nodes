"""flux-mask-refine 服务端算子注册文件：注册 flux.mask_refine
"""
import os
import random
import time

import torch

from app import execution, preview

FLUX_DIR = os.environ.get("FLUX_DIR", "/models/flux")


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
            return kw

        refined_big = i2i(
            prompt_embeds=pe, pooled_prompt_embeds=pooled,
            image=crop_img_big, strength=float(denoise),
            num_inference_steps=steps, guidance_scale=float(guidance),
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=_cb,
            callback_on_step_end_tensor_inputs=[],
        ).images[0]

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

        return {"image": out_img}
