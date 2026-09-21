"""latent-set-noise-mask：给 latent 包裹 noise_mask（局部重绘/inpaint 核心）"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("latent.set_noise_mask", {
        "latent": ref(param("latent")),
        "mask": ref(param("mask")),
    }, emit_keys=['latent'], timeout=600)


if __name__ == "__main__":
    main()
