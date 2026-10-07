"""vae-decode 节点的服务端算子插件（随节点包自注册）。

SD1.x/SDXL VAE Decode：latent → PIL 图像，委托 app.ops.vae_decode。
"""
from app import ops


def register(registry):
    @registry.register(
        "sd.vae.decode",
        inputs={"vae": "VAE", "latent": "LATENT"},
        outputs={"image": "IMAGE"},
        description="VAE Decode：latent → PIL 图像")
    def _op_vae_decode(vae, latent):
        return ops.vae_decode(vae, latent)
