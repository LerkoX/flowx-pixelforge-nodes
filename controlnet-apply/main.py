"""controlnet-apply：ControlNet 捆图（controlnet.apply_masked）。

ControlNetModel + hint 图（边缘/姿态线稿）捆绑为 CONTROL，喂 sample 的
control 端口；strength=残差强度（0 ≡ 关闭），start/end_percent 为生效步窗口。
v1.2.0：可选输入 control（上一个 apply 的 CONTROL 输出）——非空时串成
多 ControlNet 链（sample 内逐路前向、残差×各自 strength 求和注入）。
v1.3.0：可选输入 mask（灰度蒙版，白=重绘区）——非空时先把 hint 图蒙版区
烧黑再捆绑（control_v11p_sd15_inpaint 的 hint 约定）；走族版本化算子
sd.controlnet.apply_masked@<hash>，与旧算子并存。
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
    mask = param("mask", "")
    if mask:
        inputs["mask"] = ref(mask)
    chained = param("control", "")
    if chained:
        inputs["control"] = ref(chained)
    run_op("sd.controlnet.apply_masked", inputs, emit_keys=["control"])


if __name__ == "__main__":
    main()
