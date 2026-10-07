"""vae-encode 节点的服务端算子插件（随节点包自注册）。

SD1.x/SDXL VAE Encode：PIL 图像 → latent（图生图入口），委托 app.ops.vae_encode。
"""
from app import ops


def register(registry):
    @registry.register(
        "sd.vae.encode",
        inputs={"vae": "VAE", "image": "IMAGE"},
        outputs={"latent": "LATENT"},
        description="VAE Encode：PIL 图像 → latent（图生图入口，配 sd.sample 的 denoise<1 使用）")
    def _op_vae_encode(vae, image):
        return ops.vae_encode(vae, image)
