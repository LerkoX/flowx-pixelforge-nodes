"""preprocess-face：人脸裁剪（Wan-Animate face_video 契约，服务端实现）。

约定（算子全面插件化后）：server_op.py 随节点包自注册到推理服务
（ensure_plugin，hash 幂等）；族文件单一事实源在 _common/server_ops/。

算子 preprocess.face：逐帧检测最大人脸 → 按 pad 外扩成方形 → 裁到 out_size²。
给 wan-animate 的 face_video 用；IMAGE 批次输入时逐帧处理（视频帧序列等价）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("preprocess.face", {
        "image": ref(param("image")),
        "pad": param("pad", 0.4, cast=float),
        "out_size": param("out_size", 512, cast=int),
    }, emit_keys=["image"], timeout=600, input_media=("image",))


if __name__ == "__main__":
    main()
