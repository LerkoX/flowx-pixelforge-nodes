"""image-person-mask：SAM3 文本提示图像分割（精准抠头/抠人 → 灰度蒙版）。"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("seg.sam3", {
        "image": ref(param("image")),
        "prompt": param("prompt", "person"),
        "obj_pick": param("obj_pick", "all"),
        "threshold": float(param("threshold", "0.5")),
        "dilate": int(param("dilate", "0")),
    }, emit_keys=["mask", "count", "coverage", "width", "height"],
        input_media=("image",))


if __name__ == "__main__":
    main()
