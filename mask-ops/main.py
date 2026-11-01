"""mask-ops：蒙版布尔运算（union/subtract/intersect/xor）。

典型用法：从正向区域蒙版中扣除无需处理的部位（subtract），
或把多个来源的蒙版合并（union）。算子 seg.mask_ops 由 seg.py 族提供，
随节点包自注册（ensure_plugin 走 run_op 内部）。"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("seg.mask_ops", {
        "mask_a": ref(param("mask_a")),
        "mask_b": ref(param("mask_b")),
        "op": param("op", "subtract"),
    }, emit_keys=["mask", "width", "height", "coverage"])


if __name__ == "__main__":
    main()
