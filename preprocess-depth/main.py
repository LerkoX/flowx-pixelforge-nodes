"""preprocess-depth：MiDaS 单目深度估计（服务端插件）。

controlnet_aux MidasDetector（dpt_hybrid）：灰度深度图（近亮远暗），
供 ControlNet depth 控制采样作 hint（锁构图/空间层次）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("preprocess.depth", {
        "image": ref(param("image")),
        "detect_resolution": int(param("detect_resolution", "512")),
    }, emit_keys=["image"])


if __name__ == "__main__":
    main()
