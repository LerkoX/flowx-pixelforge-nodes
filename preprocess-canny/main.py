"""preprocess-canny：canny 线稿提取（真 cv2.Canny，服务端实现）。

约定（第三档批次 2 起）：preprocess.canny 的服务端实现以推理服务
plugins/preprocess_ops.py 为唯一来源，本节点包不再自注册该算子的实现，
仅做存在性校验 + 调用。算子缺失时给出明确报错（联系服务端部署插件）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("preprocess.canny", {
        "image": ref(param("image")),
        "low_threshold": param("low_threshold", 100, cast=int),
        "high_threshold": param("high_threshold", 200, cast=int),
    }, emit_keys=["image"], check_exists=True, timeout=600)


if __name__ == "__main__":
    main()
