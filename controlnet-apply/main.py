"""controlnet-apply：ControlNet 捆图（controlnet.apply）。

ControlNetModel + hint 图（边缘/姿态线稿）捆绑为 CONTROL，喂 sample 的
control 端口；strength=残差强度（0 ≡ 关闭），start/end_percent 为生效步窗口。
v1.2.0：可选输入 control（上一个 apply 的 CONTROL 输出）——非空时串成
多 ControlNet 链（sample 内逐路前向、残差×各自 strength 求和注入）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    inputs = {
        "control_net": ref(param("control_net")),
        "image": ref(param("image")),
        "strength": param("strength", 1.0, cast=float),
        "start_percent": param("start_percent", 0.0, cast=float),
        "end_percent": param("end_percent", 1.0, cast=float),
    }
    chained = param("control", "")
    if chained:
        inputs["control"] = ref(chained)
    run_op("sd.controlnet.apply", inputs, emit_keys=["control"])


if __name__ == "__main__":
    main()
