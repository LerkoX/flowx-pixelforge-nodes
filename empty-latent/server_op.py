"""empty-latent 节点的服务端算子插件（随节点包自注册）。

SD1.x/SDXL 空 latent 创建：委托 app.ops.latent_empty。
"""
from app import ops


def register(registry):
    @registry.register(
        "sd.latent.empty",
        inputs={"width": "INT", "height": "INT", "batch_size": "INT"},
        outputs={"latent": "LATENT"},
        description="Empty Latent Image：按宽高/batch 创建零 latent（均可省略，默认 512x512x1）")
    def _op_latent_empty(width=512, height=512, batch_size=1):
        return ops.latent_empty(width, height, batch_size)
