"""face-restore：人脸修复（服务端插件 face_restore.py）。

insightface 检测 → spandrel 修复（GFPGAN/RestoreFormer）→ 羽化贴回；
权重从 FACERESTORE_MODELS_DIR 本地加载，只改人脸区域。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("face.restore", {
        "image": ref(param("image")),
        "model": param("model", "gfpgan"),
        "upscale": float(param("upscale", "1.0")),
        "padding": float(param("padding", "0.3")),
        "feather": int(param("feather", "16")),
        "only_center": param("only_center", "true"),
        "max_faces": int(param("max_faces", "1")),
        "det_thresh": float(param("det_thresh", "0.5")),
    }, emit_keys=["image", "faces", "width", "height"], input_media=("image",))


if __name__ == "__main__":
    main()
