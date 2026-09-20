"""sd3-txt2img 节点的服务端算子插件（随节点包自注册，经 POST /admin/plugins 上传）。

SD3.5 文生图（MMDiT 新架构，dev-plan 9.6）：管道全包——prompt 直接进 pipe、
IMAGE 直接出，采样循环由 diffusers 承接（flow matching 细节不进本仓库）。
不与 SD1.x 的 clip.encode/latent 体系混用（SD3 是 16ch latent + pooled embed，
形状不同，MVP 不拆 COND）。

依赖服务端核心原语：app.ops 的 resolve_pipe / exec_device_of；
进度/预览/取消经 app.execution + app.preview 接线（16ch latent 无法投影成图，
预览用进度卡片，同 video.sample 的做法）。

模型经 checkpoint.load 加载（核心算子；默认弃 T5-XXL，CLIP-L/G 保底，
use_t5 旋钮可开）。分钟级任务——调用方应经异步 job（POST /jobs）执行。
"""
import inspect
import random
import time

import torch

from app import execution, ops, preview

SD3_CLASS = "StableDiffusion3Pipeline"


def register(registry):
    @registry.register(
        "sd3.txt2img",
        inputs={"model": "MODEL", "prompt": "STRING", "negative_prompt": "STRING",
                "width": "INT", "height": "INT", "steps": "INT", "cfg": "FLOAT",
                "seed": "INT", "preview_every": "INT"},
        outputs={"image": "IMAGE", "seed": "INT"},
        description="SD3.5 文生图（MMDiT 新架构，节点插件提供）：管道全包，prompt 直进、"
                    "IMAGE 直出。模型经 checkpoint.load 加载（默认弃 T5-XXL）。"
                    "分钟级任务，请经 POST /jobs 异步执行；preview_every>0 时进度卡片帧"
                    "留在 GET /preview/{job_id}；/interrupt 可取消")
    def _op_sd3_txt2img(model, prompt="", negative_prompt="", width=512,
                        height=512, steps=28, cfg=4.5, seed=-1, preview_every=1):
        pipe, patches = ops.resolve_pipe(model)
        cls_name = type(pipe).__name__
        if cls_name != SD3_CLASS:
            raise ValueError(
                f"sd3.txt2img 只接受 {SD3_CLASS} 模型（当前 {cls_name}）；"
                f"SD1.x/SDXL 请走 sample 算子")
        if patches:
            raise ValueError("SD3 LoRA 本期不支持（MVP 边界），请直连未打补丁的 MODEL")
        if width % 16 or height % 16:
            raise ValueError("SD3 要求 width/height 为 16 的倍数（VAE 8x + patchify 2x）")

        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device=ops.exec_device_of(pipe)).manual_seed(seed)

        sig = inspect.signature(pipe.__call__).parameters
        candidates = {
            "prompt": prompt or None,
            "negative_prompt": negative_prompt or None,
            "width": width, "height": height,
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "generator": gen,
            "output_type": "pil",
        }
        call = {k: v for k, v in candidates.items() if k in sig and v is not None}

        def on_step_end(p, i, t, cb_kwargs):
            execution.check_cancelled()  # 取消检查点：抛 JobCancelled 即中断
            job = execution.current()
            if job is not None:
                job.set_progress(i + 1, steps)  # 异步 job 进度（无需预览也上报）
                if preview_every > 0:
                    try:
                        rec = preview.recorder_for(job.id, preview_every)
                        if rec.want(i, steps):
                            # 16ch latent 无法廉价投影，录纯渲染进度卡片帧
                            rec.push_pil(preview.progress_card(i + 1, steps),
                                         (i + 1) / steps)
                    except Exception as e:
                        print(f"[sd3.txt2img] preview failed (ignored): {e}",
                              flush=True)
            return cb_kwargs

        if "callback_on_step_end" in sig:
            call["callback_on_step_end"] = on_step_end
        else:
            print("[sd3.txt2img] 管道不支持 callback_on_step_end："
                  "进度上报/取消降级为任务边界（粗粒度）", flush=True)

        t0 = time.time()
        out = pipe(**call)
        image = out.images[0]
        print(f"[sd3.txt2img] done in {time.time()-t0:.1f}s "
              f"size={image.size} seed={seed}", flush=True)
        return {"image": image, "seed": seed}
