"""vae-decode-video：3D latent → VIDEO（视频 VAE 解码）"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("vae.decode_video", {
        "vae": ref(param("vae")),
        "latents": ref(param("latents")),
        "num_frames": param("num_frames", 0, cast=int),
        "decode_chunk_size": param("decode_chunk_size", 14, cast=int),
        "force_fp32": param("force_fp32", True, cast=lambda v: str(v).lower() not in ("false","0","no")),
        "fps": param("fps", 24, cast=int),
    }, emit_keys=['video'], timeout=3600)


if __name__ == "__main__":
    main()
