"""video-sample-latent：SVD 视频采样只出 latent 不解码，输出 3D latent 供 vae-decode-video 接力；分钟级任务。preview_every>0 时画布实时预览采样帧"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("video.sample_latent", {
        "model": ref(param("model")),
        "prompt": param("prompt", ""),
        "neg_prompt": param("neg_prompt", ""),
        "image": ref(param("image")) if param("image", "") else None,
        "width": param("width", 832, cast=int),
        "height": param("height", 480, cast=int),
        "num_frames": param("num_frames", 121, cast=int),
        "fps": param("fps", 24, cast=int),
        "steps": param("steps", 50, cast=int),
        "cfg": param("cfg", 5.0, cast=float),
        "seed": param("seed", -1, cast=int),
        "preview_every": param("preview_every", 0, cast=int),
    }, emit_keys=['latent', 'seed'], timeout=7200)


if __name__ == "__main__":
    main()
