"""sd3-empty-latent 节点的服务端算子插件（随节点包自注册）。

SD3.5 空 Latent（对标 ComfyUI Empty Latent Image）：16 通道（SD1.x 为 4），
宽高须为 16 的倍数（VAE 8x 下采样 + patchify 2x）。噪声在 sd3.sample 采样时注入。
"""
import torch

from app import ops

SD3_LATENT_CHANNELS = 16


def register(registry):
    @registry.register(
        "sd3.latent.empty",
        inputs={"width": "INT", "height": "INT", "batch_size": "INT"},
        outputs={"latent": "LATENT", "info": "STRING"},
        description="SD3.5 Empty Latent：按宽高/batch 创建 16 通道零 latent"
                    "（默认 512x512x1；宽高须 16 的倍数）")
    def _op_sd3_latent_empty(width=512, height=512, batch_size=1):
        if width % 16 or height % 16:
            raise ValueError(
                "SD3 要求 width/height 为 16 的倍数（VAE 8x + patchify 2x）")
        latent = torch.zeros((batch_size, SD3_LATENT_CHANNELS,
                              height // 8, width // 8),
                             dtype=torch.float16, device="cuda")
        info = f"latent={tuple(latent.shape)} dtype=fp16"
        print(f"[sd3.latent.empty] {info}", flush=True)
        return {"latent": latent, "info": info}
