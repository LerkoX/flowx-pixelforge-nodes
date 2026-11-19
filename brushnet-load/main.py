"""brushnet-load：BrushNet 分支权重加载（sd.brushnet.load）。

输出 CONTROL_NET 句柄，经 brushnet-apply 捆条件（masked image + mask）
后喂 ksampler-brushnet 采样节点。
"""
from executor_base import run_op
from flowx_client import param


def main():
    run_op("sd.brushnet.load", {
        "name": param("name"),
        "dtype": param("dtype", "auto"),
    }, emit_keys=["control_net"])


if __name__ == "__main__":
    main()
