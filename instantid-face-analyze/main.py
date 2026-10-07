"""instantid-face-analyze：insightface 人脸检测+身份特征提取（服务端插件）。

antelopev2（onnxruntime CPU，不占显存）：检测人脸、512 维身份特征、
5 关键点控制图（黑底五色，InstantID ControlNet 的 controlnet_cond）。
canvas_width/canvas_height（默认 0）：给出后 kps 直接画在该尺寸画布上
（等比缩放+居中，保人脸几何）——参考图比例与生成比例不一致时填生成
画布尺寸，否则控制图被非均匀压缩，生成人脸会变扁/变长。
canvas_mode=fit(letterbox)/cover(铺满裁溢出，人脸保持原尺度，身份优选)。
输出 face（身份特征包，供 instantid-apply）+ kps（关键点图，可直接预览核对）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("face.analyze", {
        "image": ref(param("image")),
        "face_index": param("face_index", "-1", cast=int),
        "canvas_width": param("canvas_width", "0", cast=int),
        "canvas_height": param("canvas_height", "0", cast=int),
        "canvas_mode": param("canvas_mode", "fit"),
    }, emit_keys=["face", "kps"], check_exists=True)


if __name__ == "__main__":
    main()
