"""preprocess-lineart：线稿提取（三档）（服务端插件）。

controlnet_aux 预处理器：IMAGE → 黑底白线/彩图 hint（尺寸=输入尺寸），
供 ControlNet 控制采样；CPU 推理，权重本地加载。
"""
from executor_base import run_op
from flowx_client import param, ref


def _bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def main():
    run_op("preprocess.lineart", {
        "image": ref(param("image")),
        "variant": param("variant", "lineart"),
        "detect_resolution": int(param("detect_resolution", "512")),
        "coarse": _bool(param("coarse", "false")),
    }, emit_keys=["image"])


if __name__ == "__main__":
    main()
