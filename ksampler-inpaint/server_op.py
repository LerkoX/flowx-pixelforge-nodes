"""ksampler-inpaint 节点的服务端算子插件。

在现有 app.ops.sample 采样循环外包一层：检测 9 通道 SD1.x Inpaint UNet 后，
临时给 UNet.forward 拼接 [noisy_latents, mask, masked_image_latents]。
这样不改核心 sample，也能保留 ControlNet / LoRA / 分段采样 / 预览等既有能力。
"""
import torch

from app import execution, ops, preview


def _match_batch(x, batch):
    if x.shape[0] == batch:
        return x
    if x.shape[0] == 1 or batch % x.shape[0] == 0:
        n = batch // x.shape[0]
        return x.repeat(n, *([1] * (x.dim() - 1)))
    raise ValueError(f"inpaint 条件 batch 无法对齐: {tuple(x.shape)} -> {batch}")


def _call_sample(model, pos, neg, latent, seed, steps, cfg, sampler_name,
                 scheduler, denoise, start_at_step, end_at_step, add_noise,
                 control, on_step):
    return ops.sample(model, pos, neg, latent, seed, steps, cfg,
                      sampler_name, scheduler, denoise, start_at_step,
                      end_at_step, add_noise, control=control,
                      preview_cb=on_step,
                      interrupt_check=execution.check_cancelled)


def register(registry):
    @registry.register(
        "sd.sample_inpaint",
        inputs={"model": "MODEL", "pos": "COND", "neg": "COND", "latent": "LATENT",
                "masked_latent": "LATENT",
                "seed": "INT", "steps": "INT", "cfg": "FLOAT",
                "sampler_name": "STRING", "scheduler": "STRING", "denoise": "FLOAT",
                "start_at_step": "INT", "end_at_step": "INT", "add_noise": "BOOL",
                "control": "CONTROL", "preview_every": "INT"},
        outputs={"latent": "LATENT", "seed": "INT"},
        description="SD1.x Inpaint KSampler：要求 latent 带 noise_mask；9 通道 UNet "
                    "按 diffusers 原生语义先在像素空间置黑 mask 区、再 VAE 编码 "
                    "masked_image_latents，其余语义同 sd.sample")
    def _op_sample_inpaint(model, pos, neg, latent, masked_latent, seed=-1, steps=20, cfg=7.0,
                           sampler_name="euler", scheduler="normal", denoise=1.0,
                           start_at_step=0, end_at_step=0, add_noise=True,
                           control=None, preview_every=1):
        pipe, _patches = ops.resolve_pipe(model)
        hint = "sd15"
        if preview_every > 0:
            hint = preview.model_hint_of(pipe)

        def on_step(latents, i, total):
            job = execution.current()
            if job is None:
                return
            job.set_progress(i + 1, total)
            if preview_every > 0:
                rec = preview.recorder_for(job.id, preview_every, hint)
                if rec.want(i, total):
                    rec.push(latents, (i + 1) / total)

        in_ch = int(getattr(pipe.unet.config, "in_channels", 4) or 4)
        if in_ch != 9:
            print(f"[sample-inpaint] unet.in_channels={in_ch}，退回普通 sd.sample",
                  flush=True)
            return _call_sample(model, pos, neg, latent, seed, steps, cfg,
                                sampler_name, scheduler, denoise, start_at_step,
                                end_at_step, add_noise, control, on_step)

        base_samples, noise_mask = ops.resolve_latent(latent)
        if noise_mask is None:
            raise ValueError(
                "sd.sample_inpaint: 9 通道 inpaint UNet 需要 latent.set_noise_mask "
                "提供 mask（白=重绘区域）")
        device = ops.exec_device_of(pipe)
        base_samples = base_samples.to(device=device, dtype=pipe.unet.dtype)
        noise_mask = noise_mask.to(device=device, dtype=pipe.unet.dtype)

        # masked_image_latents 必须由 vae-encode-inpaint-masked 提前计算：
        # 先在线图像素空间置黑 mask 区，再 VAE 编码。这里直接消费对象，避免在
        # KSampler 阶段（UNet/CN 已常驻）再做一次高分辨率 VAE encode 顶爆显存。
        masked_image_latents, extra_mask = ops.resolve_latent(masked_latent)
        if extra_mask is not None:
            raise ValueError("masked_latent 必须是裸 latent，不能带 noise_mask")
        masked_image_latents = masked_image_latents.to(device=device,
                                                       dtype=pipe.unet.dtype)
        if masked_image_latents.shape != base_samples.shape:
            raise ValueError(
                f"masked_latent 尺寸 {tuple(masked_image_latents.shape)} 与 "
                f"输入 latent {tuple(base_samples.shape)} 不符")
        print("[sample-inpaint] 9ch UNet: concat latent + mask + "
              "masked_image_latents（前置像素置黑 VAE 编码）", flush=True)

        orig_forward = pipe.unet.forward

        def inpaint_forward(sample, timestep, encoder_hidden_states=None, **kwargs):
            b = sample.shape[0]
            m = _match_batch(noise_mask, b).to(device=sample.device,
                                             dtype=sample.dtype)
            mi = _match_batch(masked_image_latents, b).to(device=sample.device,
                                                          dtype=sample.dtype)
            if m.shape[-2:] != sample.shape[-2:] or mi.shape[-2:] != sample.shape[-2:]:
                raise ValueError(
                    f"inpaint 条件尺寸不符: latent={tuple(sample.shape)} "
                    f"mask={tuple(m.shape)} masked={tuple(mi.shape)}")
            return orig_forward(torch.cat([sample, m, mi], dim=1), timestep,
                                encoder_hidden_states=encoder_hidden_states,
                                **kwargs)

        pipe.unet.forward = inpaint_forward
        try:
            return _call_sample(model, pos, neg, latent, seed, steps, cfg,
                                sampler_name, scheduler, denoise, start_at_step,
                                end_at_step, add_noise, control, on_step)
        finally:
            pipe.unet.forward = orig_forward
