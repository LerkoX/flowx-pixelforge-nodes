"""preprocess-openpose：openpose 姿态骨架提取（服务端插件）。

controlnet_aux OpenposeDetector：全身骨架（默认），可选手部/面部细节；
产出骨架叠加图，供 ControlNet openpose 控制采样作 hint。
"""
from executor_base import run_op
from flowx_client import param, ref


def _bool(v):
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def main():
    run_op("preprocess.openpose", {
        "image": ref(param("image")),
        "include_hand": _bool(param("include_hand", "false")),
        "include_face": _bool(param("include_face", "false")),
    }, emit_keys=["image"], check_exists=True)


if __name__ == "__main__":
    main()
