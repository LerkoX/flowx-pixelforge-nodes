"""preprocess-canny：canny 线稿提取（真 cv2.Canny，服务端实现）。

约定（算子全面插件化后）：server_op.py 随节点包自注册到推理服务
（ensure_plugin，hash 幂等）；族文件单一事实源在 _common/server_ops/。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("preprocess.canny", {
        "image": ref(param("image")),
        "low_threshold": param("low_threshold", 100, cast=int),
        "high_threshold": param("high_threshold", 200, cast=int),
    }, emit_keys=["image"], timeout=600)


if __name__ == "__main__":
    main()
