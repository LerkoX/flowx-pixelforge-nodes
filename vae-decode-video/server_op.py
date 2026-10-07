"""vae-decode-video 节点的服务端算子插件（随节点包自注册）。

视频 VAE 解码：3D latent → VIDEO，委托 app.ops.vae_decode_video
（force_fp32 防 Pascal fp16 解码过曝；decode_chunk_size 分块控显存峰值）。
"""
from app import ops


def register(registry):
    @registry.register(
        "video.vae.decode",
        inputs={"vae": "VAE", "latents": "LATENT", "num_frames": "INT",
                "decode_chunk_size": "INT", "force_fp32": "BOOL", "fps": "INT"},
        outputs={"video": "VIDEO"},
        description="VAE Decode（视频）：3D latent → VIDEO。force_fp32 防 Pascal fp16 "
                    "解码过曝/亮度漂移；decode_chunk_size 分块控制显存峰值（SVD 有效）")
    def _op_vae_decode_video(vae, latents, num_frames=0, decode_chunk_size=14,
                             force_fp32=True, fps=24):
        return ops.vae_decode_video(vae, latents, num_frames, decode_chunk_size,
                                    force_fp32, fps)
