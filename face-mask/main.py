"""face-mask：人脸 bbox → feathered 矩形蒙版 + crop 框（服务端插件）。

insightface 检测人脸 bbox，外扩+8 倍数对齐后输出 mask 与坐标，
供 image-crop / image-composite 直接绑定（FaceDetailer 局部精修编排入口）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("face.mask", {
        "image": ref(param("image")),
        "face_index": int(param("face_index", "-1")),
        "det_thresh": float(param("det_thresh", "0.2")),
        "expand": float(param("expand", "0.6")),
        "feather": float(param("feather", "16")),
    }, emit_keys=["mask", "x", "y", "width", "height"])


if __name__ == "__main__":
    main()
