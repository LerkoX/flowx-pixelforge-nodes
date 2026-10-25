"""flux-pulid-face：FLUX PuLID 人脸身份编码（服务端插件）。

参考图 → FACE 对象（id_embedding + 交叉注意力模块），供 flux-sampler
的 pulid 参数做 PuLID-FLUX 身份注入。人脸检测/对齐/抠脸/EVA 编码全部
在服务端完成，参考图无需预裁剪。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("flux.pulid_encode", {
        "image": ref(param("image")),
        "face_index": int(param("face_index", "-1")),
    }, emit_keys=["pulid", "info"], input_media=("image",))


if __name__ == "__main__":
    main()
