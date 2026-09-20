"""sd3-sampler 节点的服务端算子插件（随节点包自注册）。

SD3.5 采样器（对标 ComfyUI KSampler）：flow matching 手动采样循环——
调度器按底管配置新建实例（不污染常驻管道共享状态）；latent 全程保持
(b,16,h/8,w/8) 形状，patchify（patch_size=2）由 transformer 内部 PatchEmbed
处理（已对 diffusers 0.35.2 源码核实：管道 denoising loop 直传未打包 latent）。
denoise=1 文生图（空 latent 注入纯噪声起步）；denoise<1 图生图
（sd3.vae.encode 的 latent + scheduler.scale_noise 部分加噪起步）。

进度/预览/取消经 app.execution + app.preview 接线（16ch latent 无法投影成图，
预览用进度卡片）。分钟级任务——调用方应经异步 job（POST /jobs）执行。
"""
import random
import time

import torch

from app import execution, ops, preview

SD3_CLASS = "StableDiffusion3Pipeline"


def register(registry):
    @registry.register(
        "sd3.sample",
        inputs={"model": "MODEL", "pos": "COND", "neg": "COND", "latent": "LATENT",
                "seed": "INT", "steps": "INT", "cfg": "FLOAT", "denoise": "FLOAT",
                "preview_every": "INT"},
        outputs={"latent": "LATENT", "seed": "INT"},
        description="SD3.5 Sampler：flow matching 采样循环；denoise=1 文生图 / "
                    "denoise<1 图生图；preview_every>0 时进度卡片帧留在 "
                    "GET /preview/{job_id}；/interrupt 可取消")
    def _op_sd3_sample(model, pos, neg, latent, seed=-1, steps=28, cfg=4.5,
                       denoise=1.0, preview_every=1):
        pipe, patches = ops.resolve_pipe(model)
        cls_name = type(pipe).__name__
        if cls_name != SD3_CLASS:
            raise ValueError(
                f"sd3.sample 只接受 {SD3_CLASS} 模型（当前 {cls_name}）")
        if patches:
            raise ValueError("SD3 LoRA 本期不支持（MVP 边界），请直连未打补丁的 MODEL")

        from diffusers import FlowMatchEulerDiscreteScheduler

        device = ops.exec_device_of(pipe)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device=device).manual_seed(seed)

        # 调度器新建实例（from_config），不污染常驻管道的共享 scheduler 状态
        sched = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
        sched.set_timesteps(steps, device=device)
        timesteps = sched.timesteps

        # 起步 latent：denoise>=1 注入纯噪声（输入 latent 仅作形状占位，
        # flow matching 的 init 即标准正态）；denoise<1 图生图部分加噪起步
        noise = torch.randn(latent.shape, generator=gen, device=device,
                            dtype=latent.dtype)
        if denoise >= 0.999:
            latents = noise
            ts = timesteps
        else:
            start = min(int(round(steps * (1 - denoise))), steps - 1)
            ts = timesteps[start:]
            latents = sched.scale_noise(latent, timesteps[start:start + 1], noise)
            print(f"[sd3.sample] denoise={denoise} -> skip first {start} steps",
                  flush=True)
        latents = latents.to(device=device, dtype=torch.float16)

        # COND 解包：{"embeds": (b,154,4096), "pooled": (b,2048)}
        pos_emb, pos_pooled = pos["embeds"], pos["pooled"]
        neg_emb, neg_pooled = neg["embeds"], neg["pooled"]
        do_cfg = cfg > 1.0
        b = latents.shape[0]

        def on_step(i, total):
            execution.check_cancelled()  # 抛 JobCancelled 即中断
            job = execution.current()
            if job is not None:
                job.set_progress(i + 1, total)
                if preview_every > 0:
                    try:
                        rec = preview.recorder_for(job.id, preview_every)
                        if rec.want(i, total):
                            rec.push_pil(preview.progress_card(i + 1, total),
                                         (i + 1) / total)
                    except Exception as e:
                        print(f"[sd3.sample] preview failed (ignored): {e}",
                              flush=True)

        t0 = time.time()
        with torch.no_grad():
            for i, t in enumerate(ts):
                inp = torch.cat([latents] * 2) if do_cfg else latents
                timestep = t.expand(inp.shape[0])
                enc = (torch.cat([neg_emb.expand(b, -1, -1),
                                  pos_emb.expand(b, -1, -1)])
                       if do_cfg else pos_emb)
                pooled = (torch.cat([neg_pooled.expand(b, -1),
                                     pos_pooled.expand(b, -1)])
                          if do_cfg else pos_pooled)
                noise_pred = pipe.transformer(
                    hidden_states=inp, timestep=timestep,
                    encoder_hidden_states=enc, pooled_projections=pooled,
                    return_dict=False)[0]
                if do_cfg:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + cfg * (
                        noise_pred_cond - noise_pred_uncond)
                latents = sched.step(noise_pred, t, latents,
                                     return_dict=False)[0]
                print(f"[sd3.sample] step {i+1}/{len(ts)}", flush=True)
                on_step(i, len(ts))

        print(f"[sd3.sample] done in {time.time()-t0:.1f}s seed={seed}", flush=True)
        return {"latent": latents, "seed": seed}
