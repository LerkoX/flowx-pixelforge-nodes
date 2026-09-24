"""instantid-face-analyze：insightface 人脸检测+身份特征提取（服务端插件）。

antelopev2（onnxruntime CPU，不占显存）：检测人脸、512 维身份特征、
5 关键点控制图（黑底五色，InstantID ControlNet 的 controlnet_cond）。
输出 face（身份特征包，供 instantid-apply）+ kps（关键点图，可直接预览核对）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("face.analyze", {
        "image": ref(param("image")),
        "face_index": param("face_index", "-1", cast=int),
    }, emit_keys=["face", "kps"], check_exists=True)


if __name__ == "__main__":
    main()
