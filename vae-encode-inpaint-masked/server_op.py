"""vae-encode-inpaint-masked 服务端算子。

按 diffusers StableDiffusionInpaintPipeline 原生语义构造 9ch UNet 的
masked_image_latents：masked_image = image * (mask < 0.5)，然后 VAE encode。
"""
import torch

from app import ops


def register(registry):
    @registry.register(
        "sd.vae.encode_inpaint_masked",
        inputs={"vae": "VAE", "image": "IMAGE", "mask": "IMAGE"},
        outputs={"latent": "LATENT"},
        description="Inpaint masked VAE Encode：先像素空间置黑 mask 区，再编码 latent")
    def _op_vae_encode_inpaint_masked(vae, image, mask):
        pipe, _ = ops.resolve_pipe(vae)
        if getattr(pipe, "mask_processor", None) is None:
            raise ValueError("sd.vae.encode_inpaint_masked 需要 inpaint 管道（缺 mask_processor）")

        width, height = image.size
        init_image = pipe.image_processor.preprocess(image, height=height, width=width)
        mask_condition = pipe.mask_processor.preprocess(mask, height=height, width=width)
        masked_image = init_image * (mask_condition < 0.5)

        device = ops.exec_device_of(pipe)
        masked_image = masked_image.to(device=device, dtype=pipe.vae.dtype)
        gen = torch.Generator(device=device).manual_seed(0)
        pipe.vae.enable_tiling()
        try:
            with torch.no_grad():
                if hasattr(pipe, "_encode_vae_image"):
                    latents = pipe._encode_vae_image(masked_image, generator=gen)
                else:
                    latents = (pipe.vae.encode(masked_image).latent_dist.sample(generator=gen)
                               * pipe.vae.config.scaling_factor)
        finally:
            pipe.vae.disable_tiling()
        latents = latents.to(dtype=torch.float16)
        print(f"[vae.encode-inpaint-masked] image={width}x{height} -> "
              f"latent={tuple(latents.shape)} mean={latents.float().mean().item():.4f}",
              flush=True)
        return {"latent": latents}
