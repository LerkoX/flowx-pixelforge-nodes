"""brushnet-apply：BrushNet 捆条件（sd.brushnet.apply）。

官方语义：masked_image = image × (1-mask)（重绘区置黑）→ VAE 编码 4ch
+ mask 插值到 latent 尺寸 1ch = 5ch 条件，与 BrushNet 模型打包成 CONTROL。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("sd.brushnet.apply", {
        "control_net": ref(param("control_net")),
        "vae": ref(param("vae")),
        "image": ref(param("image")),
        "mask": ref(param("mask")),
        "scale": param("scale", 1.0, cast=float),
        "start_percent": param("start_percent", 0.0, cast=float),
        "end_percent": param("end_percent", 1.0, cast=float),
    }, emit_keys=["control"], input_media=("image",))


if __name__ == "__main__":
    main()
