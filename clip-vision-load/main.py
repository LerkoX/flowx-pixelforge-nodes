"""clip-vision-load：加载 CLIP Vision 图像编码器（IPAdapter 参考图编码器）"""
from executor_base import run_op
from flowx_client import param


def main():
    run_op("clip_vision.load", {
        "name": param("name"),
    }, emit_keys=["clip_vision"], timeout=600)


if __name__ == "__main__":
    main()
