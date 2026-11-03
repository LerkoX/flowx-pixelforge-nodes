"""zero123 节点的服务端算子插件（随节点包自注册，经 POST /admin/plugins 上传）。

Stable Zero123 单图新视角合成：IMAGE -> IMAGE（新视角 RGB 图）。
- 权重：ashawkey/stable-zero123-diffusers（公开，HF 上有完整 diffusers 目录，
  含 unet/vae/image_encoder/scheduler/clip_camera_projection 及**自带 pipeline 代码**
  clip_camera_projection/zero123.py——diffusers 主包已移除 StableZero123Pipeline，
  故用该 repo 的 custom_pipeline 加载）。放 ZERO123_MODELS_DIR
  （默认 /models/stable-zero123/），**零网络访问**（local_files_only）。
- 依赖：diffusers（custom_pipeline + trust_remote_code）/ transformers / torch，
  一律函数内懒加载。
- 显存：8GB 卡用 fp16 + enable_model_cpu_offload（组件按需搬 GPU）。vram=after_op
  用完即释放（torch.cuda.empty_cache），与 SD 等模型错峰共存。
- 视角：elevation=仰角（±90°）/ azimuth=方位角（±180°，0=正面/90=右/180=背/-90=左）
  / distance=相机距离（默认 0）。
- 输入建议去背（Zero123 训练数据为白底物体）；RGBA 会合成到白底后再推理。
"""
import os
import threading

ZERO123_DIR = os.environ.get("ZERO123_MODELS_DIR", "/models/stable-zero123")

_pipe = None
_lock = threading.Lock()


def _load():
    """Zero123Pipeline（进程级单例）：fp16 + cpu offload。"""
    global _pipe
    with _lock:
        if _pipe is None:
            import torch
            from diffusers import DiffusionPipeline
            if not torch.cuda.is_available():
                raise RuntimeError("novelview.zero123: 需要 CUDA（Stable Zero123 "
                                   "为 GPU 推理）")
            # repo 内 clip_camera_projection/zero123.py 定义了 Zero123Pipeline +
            # CLIPCameraProjection。custom_pipeline 必须指本地 .py 文件（否则会被当
            # community pipeline 名去 HF 下载）；trust_remote_code 允许执行该自定义代码。
            pipe = DiffusionPipeline.from_pretrained(
                ZERO123_DIR,
                custom_pipeline=os.path.join(ZERO123_DIR, "clip_camera_projection",
                                             "zero123.py"),
                trust_remote_code=True, torch_dtype=torch.float16,
                local_files_only=True)
            # 该自定义 pipeline 自带 enable_sequential_cpu_offload（用 accelerate
            # cpu_offload），没有 model_cpu_offload_seq 属性，不能用基类的
            # enable_model_cpu_offload。8GB 卡逐层 offload 最稳。
            pipe.enable_sequential_cpu_offload()
            # clip_camera_projection（772→768 Linear）不在该 pipeline 自带 offload
            # 清单里，留在 CPU 会与 image_encoder 的 CUDA embedding 出现 device
            # mismatch；它很小（~2.4MB），直接常驻 GPU。
            pipe.clip_camera_projection.to("cuda")
            _pipe = pipe
            print(f"[novelview.zero123] Stable Zero123 已加载：{ZERO123_DIR} "
                  f"(fp16/cpu-offload)", flush=True)
    return _pipe


def _release():
    global _pipe
    with _lock:
        _pipe = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def register(registry):
    @registry.register(
        "novelview.zero123",
        inputs={"image": "IMAGE", "elevation": "FLOAT", "azimuth": "FLOAT",
                "distance": "FLOAT", "steps": "INT", "guidance_scale": "FLOAT",
                "vram": "STRING"},
        outputs={"image": "IMAGE", "width": "INT", "height": "INT"},
        description="Stable Zero123 单图新视角合成（IMAGE -> IMAGE）。"
                    "elevation=仰角(±90°)；azimuth=方位角(±180°，"
                    "0=正面/90=右/180=背/-90=左)；distance=相机距离(默认0)；"
                    "steps=采样步数(默认50)；guidance_scale=引导强度(默认3.0)；"
                    "vram=keep/after_op。权重在 ZERO123_MODELS_DIR"
                    "（默认 /models/stable-zero123/），建议输入已去背"
                    "（RGBA 自动合成白底）")
    def _op_novelview_zero123(image, elevation=0.0, azimuth=0.0, distance=0.0,
                              steps=50, guidance_scale=3.0, vram="after_op"):
        from PIL import Image
        if not isinstance(image, Image.Image):
            raise ValueError(f"novelview.zero123: image 需为 IMAGE 对象，got "
                             f"{type(image).__name__}")
        # RGBA（去背图）合成到白底；RGB 直接用
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.getchannel("A"))
            src = bg
        else:
            src = image.convert("RGB")
        # Zero123 模型原生 256×256：_get_latent_model_input 直接把输入图
        # vae.encode 后与噪声 latent 在 channel 维 concat，要求两者空间尺寸一致
        # （256/8=32），故输入图必须缩到 256×256（模型固有限制，后续可超分）。
        src = src.resize((256, 256), Image.LANCZOS)
        pipe = _load()
        result = pipe(
            src,
            elevation=float(elevation),
            azimuth=float(azimuth),
            distance=float(distance),
            num_inference_steps=int(steps),
            guidance_scale=float(guidance_scale),
        ).images[0]
        out = result.convert("RGB")
        print(f"[novelview.zero123] elevation={elevation} azimuth={azimuth} "
              f"distance={distance} steps={steps} -> {out.width}x{out.height}",
              flush=True)
        if str(vram) == "after_op":
            _release()
        return {"image": out, "width": out.width, "height": out.height}
