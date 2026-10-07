"""video.sample / video.sample_latent 算子族（video-gen / video-sample-latent 节点共享）。

视频采样算子族：video.sample（采样+解码一体）与 video.sample_latent
（只采样出 3D latent，交 video.vae.decode 接力）。两个节点共享本文件——
video-gen 的 latent 输出模式也调 video.sample_latent，而插件上传是单文件
机制，同族算子同文件可避免跨文件重名 409（同 flux 家族模式）。

分钟级任务，调用方应经异步 job（POST /jobs）执行；进度/预览/取消经
app.execution + app.preview 接线（3D latent 无法廉价投影，预览用进度卡片）。
"""
from app import execution, ops, preview


def _video_on_step(preview_every):
    """视频采样逐步回调：job 进度上报 + 进度卡片预览（3D latent 无法廉价投影）。"""
    def on_step(latents, i, total):
        job = execution.current()
        if job is None:
            return  # 同步 /op 调用无 job 上下文：不录预览
        job.set_progress(i + 1, total)  # 异步 job 进度（轮询通道）
        if preview_every > 0:
            rec = preview.recorder_for(job.id, preview_every)
            if rec.want(i, total):
                # 视频 3D latent 无法廉价投影成图，录纯渲染的进度卡片帧
                rec.push_pil(preview.progress_card(i + 1, total), (i + 1) / total)
    return on_step


def register(registry):
    @registry.register(
        "video.sample",
        inputs={"model": "MODEL", "prompt": "STRING", "neg_prompt": "STRING",
                "image": "IMAGE", "width": "INT", "height": "INT",
                "num_frames": "INT", "fps": "INT", "steps": "INT", "cfg": "FLOAT",
                "seed": "INT", "decode_chunk_size": "INT",
                "preview_every": "INT"},
        outputs={"video": "VIDEO", "seed": "INT"},
        description="Video Sample：文/图生视频，按管道签名自适应（Wan TI2V 文本+可选首帧 / "
                    "SVD 纯图生视频，cfg 映射 min/max_guidance_scale，fps 进采样条件）。"
                    "分钟级任务，请经 POST /jobs 异步执行；进度经 job 轮询上报，"
                    "preview_every>0 时进度卡片帧留在 GET /preview/{job_id}；"
                    "/interrupt 可取消")
    def _op_video_sample(model, prompt="", neg_prompt="", image=None,
                         width=832, height=480, num_frames=121, fps=24,
                         steps=50, cfg=5.0, seed=-1, decode_chunk_size=0,
                         preview_every=0):
        return ops.video_sample(model, prompt, neg_prompt, image, width, height,
                                num_frames, fps, steps, cfg, seed,
                                decode_chunk_size,
                                preview_cb=_video_on_step(preview_every),
                                interrupt_check=execution.check_cancelled)

    @registry.register(
        "video.sample_latent",
        inputs={"model": "MODEL", "prompt": "STRING", "neg_prompt": "STRING",
                "image": "IMAGE", "width": "INT", "height": "INT",
                "num_frames": "INT", "fps": "INT", "steps": "INT", "cfg": "FLOAT",
                "seed": "INT", "preview_every": "INT"},
        outputs={"latent": "LATENT", "seed": "INT"},
        description="Video Sample（latent 模式）：只采样不解码，输出 3D latent 供 "
                    "video.vae.decode 接力——decode 精度（fp32）与分块成为流水线可调参数。"
                    "分钟级任务，请经 POST /jobs 异步执行")
    def _op_video_sample_latent(model, prompt="", neg_prompt="", image=None,
                                width=832, height=480, num_frames=121, fps=24,
                                steps=50, cfg=5.0, seed=-1, preview_every=0):
        return ops.video_sample(model, prompt, neg_prompt, image, width, height,
                                num_frames, fps, steps, cfg, seed,
                                output_type="latent",
                                preview_cb=_video_on_step(preview_every),
                                interrupt_check=execution.check_cancelled)
