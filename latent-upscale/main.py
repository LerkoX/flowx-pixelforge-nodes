"""latent-upscale：latent 空间插值放大"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("latent.upscale", {
        "latent": ref(param("latent")),
        "scale": param("scale", 2.0, cast=float),
        "width": param("width", 0, cast=int),
        "height": param("height", 0, cast=int),
        "method": param("method", "bicubic"),
    }, emit_keys=['latent'], timeout=600)


if __name__ == "__main__":
    main()
