"""sd3-vae-decode 节点的服务端算子插件（随节点包自注册）。

SD3.5 VAE 解码（对标 ComfyUI VAE Decode）：16 通道 latent → IMAGE。
SD3 的 latent 归一化与 SD1.x 不同：(latents / scaling_factor) + shift_factor
（sd3.5-medium 实测 scaling=1.5305 / shift=0.0609，从 vae config 读，不写死）。
"""
import torch

from app import ops

SD3_CLASS = "StableDiffusion3Pipeline"


def register(registry):
    @registry.register(
        "sd3.vae.decode",
        inputs={"vae": "VAE", "latent": "LATENT"},
        outputs={"image": "IMAGE"},
        description="SD3.5 VAE Decode：16ch latent → PIL 图像"
                    "（(latents/scaling)+shift，供 sd3.sample 下游）")
    def _op_sd3_vae_decode(vae, latent):
        pipe, _ = ops.resolve_pipe(vae)
        cls_name = type(pipe).__name__
        if cls_name != SD3_CLASS:
            raise ValueError(
                f"sd3.vae.decode 只接受 {SD3_CLASS} 的 VAE 视图（当前 {cls_name}）")
        latents = (latent / pipe.vae.config.scaling_factor
                   + pipe.vae.config.shift_factor)
        with torch.no_grad():
            img = pipe.vae.decode(latents.to(dtype=pipe.vae.dtype),
                                  return_dict=False)[0]
        pil = pipe.image_processor.postprocess(img, output_type="pil")
        return {"image": pil[0] if len(pil) == 1 else pil}
