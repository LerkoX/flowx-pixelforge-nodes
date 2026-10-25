"""controlnet-apply 节点的服务端算子插件（随节点包自注册）。

SD1.x/SDXL ControlNet 捆图：ControlNetModel + hint 图 → CONTROL，
委托 app.ops.controlnet_apply。可选输入 control（上一个 apply 的 CONTROL）：
串成多 ControlNet 链（list[ControlBundle]），sample 内逐路前向、残差求和。
"""
from app import ops


def register(registry):
    @registry.register(
        "sd.controlnet.apply",
        inputs={"control_net": "CONTROL_NET", "image": "IMAGE",
                "strength": "FLOAT", "start_percent": "FLOAT",
                "end_percent": "FLOAT", "control": "CONTROL"},
        outputs={"control": "CONTROL"},
        description="ControlNet Apply：ControlNetModel + hint 图（边缘/姿态等线稿，"
                    "预处理器产出）捆绑为 CONTROL；strength=残差强度（默认 1.0，"
                    "0 ≡ 关闭）；start/end_percent 为生效步窗口（默认全程）。"
                    "可选串接：control 接上一个 apply 的 CONTROL 输出，捆绑追加成"
                    "多 ControlNet 链，采样时每路各自前向、残差×各自 strength 求和"
                    "注入（ComfyUI 多 CN 语义）。暂不与 cond.set_area 组合")
    def _op_controlnet_apply(control_net, image, strength=1.0,
                             start_percent=0.0, end_percent=1.0, control=None):
        return ops.controlnet_apply(control_net, image, strength,
                                    start_percent, end_percent, control=control)
