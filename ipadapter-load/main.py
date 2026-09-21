"""ipadapter-load：加载 IPAdapter 权重（标准版/plus 版自动判别）"""
from executor_base import run_op
from flowx_client import param


def main():
    run_op("ipadapter.load", {
        "name": param("name"),
    }, emit_keys=["ipadapter"], timeout=600)


if __name__ == "__main__":
    main()
