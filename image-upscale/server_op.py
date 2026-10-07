"""image-upscale 节点的服务端算子插件（随节点包自注册）。

图像放大（hires.fix 前置）：纯 PIL 重采样，委托 app.ops.image_upscale。
"""
from app import ops


def register(registry):
    @registry.register(
        "image.upscale",
        inputs={"image": "IMAGE", "scale": "FLOAT", "width": "INT",
                "height": "INT", "method": "STRING"},
        outputs={"image": "IMAGE"},
        description="Image Upscale：按 scale 倍率（默认 2.0）或目标 width/height 放大图像，"
                    "method 支持 lanczos/bicubic/bilinear/nearest；结果对齐 8 的倍数。"
                    "hires.fix 前置：放大 → sd.vae.encode → sd.sample(denoise 0.3~0.5) 精修细节")
    def _op_image_upscale(image, scale=2.0, width=0, height=0, method="lanczos"):
        return ops.image_upscale(image, scale, width, height, method)
