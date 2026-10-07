"""ksampler 节点的服务端算子插件（随节点包自注册）。

SD1.x/SDXL KSampler：委托 app.ops.sample（sampler_name × scheduler 双参数解耦，
分段采样/ControlNet 残差注入/cond.set_area 区域混合/noise_mask 局部重绘）。
进度/预览/取消经 app.execution + app.preview 接线（异步 job 通道）。
"""
from app import execution, ops, preview


def register(registry):
    @registry.register(
        "sd.sample",
        inputs={"model": "MODEL", "pos": "COND", "neg": "COND", "latent": "LATENT",
                "seed": "INT", "steps": "INT", "cfg": "FLOAT",
                "sampler_name": "STRING", "scheduler": "STRING", "denoise": "FLOAT",
                "start_at_step": "INT", "end_at_step": "INT", "add_noise": "BOOL",
                "control": "CONTROL", "preview_every": "INT"},
        outputs={"latent": "LATENT", "seed": "INT"},
        description="KSampler：sampler_name（更新公式：euler/euler_a/ddim/lms/dpmpp_2m/"
                    "dpmpp_2m_sde/uni_pc）× scheduler（sigma 曲线：normal/karras/"
                    "exponential/beta）自由组合，组合支持性按 sampler 类能力校验；"
                    "seed/steps/cfg/sampler_name/scheduler/denoise 均有默认值；"
                    "兼容旧一体名 dpmpp_2m_karras；"
                    "分段采样（KSampler Advanced）：start_at_step>0 从该步开始（优先于 "
                    "denoise），end_at_step>0 提前停（latent 带残余噪声可接力），"
                    "add_noise=false 不加噪直接接力上一段输出；"
                    "control 可选（controlnet.apply 产物）：ControlNet 残差注入，"
                    "正/负等长时按 CFG 拼批单次前向；"
                    "COND 支持 cond.set_area 分段（区域条件混合）；"
                    "latent 支持 latent.set_noise_mask 包裹（局部重绘，mask 外混回原图）；"
                    "异步 job 执行且 preview_every>0 时，逐步把 latent 预览帧（JPEG）"
                    "留在 GET /preview/{job_id}（只留最新一帧），供 Studio 中转拉取")
    def _op_sample(model, pos, neg, latent, seed=-1, steps=20, cfg=7.0,
                   sampler_name="euler", scheduler="normal", denoise=1.0,
                   start_at_step=0, end_at_step=0, add_noise=True,
                   control=None, preview_every=1):
        hint = "sd15"
        if preview_every > 0:
            pipe, _ = ops.resolve_pipe(model)
            hint = preview.model_hint_of(pipe)

        def on_step(latents, i, total):
            job = execution.current()
            if job is None:
                return  # 同步 /op 调用无 job 上下文：不录预览（预览走异步 job 通道）
            job.set_progress(i + 1, total)  # 异步 job 进度（无需预览也上报）
            if preview_every > 0:
                rec = preview.recorder_for(job.id, preview_every, hint)
                if rec.want(i, total):
                    rec.push(latents, (i + 1) / total)

        return ops.sample(model, pos, neg, latent, seed, steps, cfg,
                          sampler_name, scheduler, denoise, start_at_step,
                          end_at_step, add_noise, control=control,
                          preview_cb=on_step,
                          interrupt_check=execution.check_cancelled)
