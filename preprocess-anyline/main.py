"""preprocess-anyline：Anyline 高精度线稿（MTEED）（服务端插件）。

controlnet_aux 预处理器：IMAGE → 黑底白线/彩图 hint（尺寸=输入尺寸），
供 ControlNet 控制采样；CPU 推理，权重本地加载。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("preprocess.anyline", {
        "image": ref(param("image")),
        "detect_resolution": int(param("detect_resolution", "1280")),
        "guassian_sigma": float(param("guassian_sigma", "2.0")),
        "intensity_threshold": int(param("intensity_threshold", "3")),
    }, emit_keys=["image"])


if __name__ == "__main__":
    main()
