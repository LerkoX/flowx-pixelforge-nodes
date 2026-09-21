"""upscale-model-apply：模型放大（image.upscale_with_model，服务端插件）。

UPSCALE_MODEL + 图像 → 放大图；tile=0 整图（默认，显存够时最快），
>0 分块（overlap 16px 消接缝，大分辨率防爆显存）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("image.upscale_with_model", {
        "upscale_model": ref(param("upscale_model")),
        "image": ref(param("image")),
        "tile": param("tile", 0, cast=int),
    }, emit_keys=["image"], check_exists=True)


if __name__ == "__main__":
    main()
