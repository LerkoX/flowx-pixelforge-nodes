"""controlnet-load：ControlNet 模型加载（controlnet.load）。

输出 CONTROL_NET 句柄，经 controlnet-apply 捆绑 hint 图后喂采样节点。
"""
from executor_base import run_op
from flowx_client import param


def main():
    run_op("controlnet.load", {
        "name": param("name"),
        "dtype": param("dtype", "auto"),
    }, emit_keys=["control_net"])


if __name__ == "__main__":
    main()
