"""sd3-vae-encode 节点的服务端算子插件（随节点包自注册）。

SD3.5 VAE 编码（对标 ComfyUI VAE Encode）：IMAGE → 16 通道 latent，
供 sd3.sample 以 denoise<1 做图生图。归一化与 decode 互逆：
(latents - shift_factor) * scaling_factor（ComfyUI VAEEncode 同语义）。
"""
import torch

from app import ops

SD3_CLASS = "StableDiffusion3Pipeline"


def register(registry):
    @registry.register(
        "sd3.vae.encode",
        inputs={"vae": "VAE", "image": "IMAGE"},
        outputs={"latent": "LATENT"},
        description="SD3.5 VAE Encode：PIL 图像 → 16ch latent"
                    "（(latents-shift)*scaling，配 sd3.sample denoise<1 使用）")
    def _op_sd3_vae_encode(vae, image):
        pipe, _ = ops.resolve_pipe(vae)
        cls_name = type(pipe).__name__
        if cls_name != SD3_CLASS:
            raise ValueError(
                f"sd3.vae.encode 只接受 {SD3_CLASS} 的 VAE 视图（当前 {cls_name}）")
        images = image if isinstance(image, list) else [image]
        tensors = [pipe.image_processor.preprocess(img) for img in images]
        img_t = torch.cat(tensors).to(device=ops.exec_device_of(pipe),
                                      dtype=pipe.vae.dtype)
        with torch.no_grad():
            latents = pipe.vae.encode(img_t).latent_dist.sample()
        latents = (latents - pipe.vae.config.shift_factor) \
            * pipe.vae.config.scaling_factor
        return {"latent": latents}
