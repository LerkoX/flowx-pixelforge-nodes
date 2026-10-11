"""matting-birefnet：BiRefNet 抠图（服务端插件 matting.py）。

IMAGE → 前景 alpha 蒙版（灰度）+ RGBA 去背图；变体 lite/main/portrait，
权重从 MATTING_MODELS_DIR 本地加载，GPU 推理。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("matting.birefnet", {
        "image": ref(param("image")),
        "variant": param("variant", "lite"),
        "resolution": int(param("resolution", "1024")),
        "use_checkpoint": param("use_checkpoint", "false"),
        "refine": param("refine", "true"),
        "refine_radius": int(param("refine_radius", "90")),
        "alpha_gamma": float(param("alpha_gamma", "1.0")),
        "invert": param("invert", "false"),
    }, emit_keys=["mask", "image", "width", "height"])


if __name__ == "__main__":
    main()
