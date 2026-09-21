"""upscale-model-load：放大模型加载（upscale_model.load，服务端插件）。

spandrel 自动识别架构（RealESRGAN/SwinIR/…），返回 UPSCALE_MODEL 句柄
（含 scale/dtype），喂 upscale-model-apply。
"""
from executor_base import run_op
from flowx_client import param


def main():
    run_op("upscale_model.load", {
        "name": param("name"),
    }, emit_keys=["upscale_model"], check_exists=True)


if __name__ == "__main__":
    main()
