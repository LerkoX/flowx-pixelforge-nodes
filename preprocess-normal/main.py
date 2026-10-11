"""preprocess-normal：法线图（NormalBae）（服务端插件）。

controlnet_aux 预处理器：IMAGE → 黑底白线/彩图 hint（尺寸=输入尺寸），
供 ControlNet 控制采样；CPU 推理，权重本地加载。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("preprocess.normal", {
        "image": ref(param("image")),
        "detect_resolution": int(param("detect_resolution", "512")),
    }, emit_keys=["image"])


if __name__ == "__main__":
    main()
