"""controlnet-apply 节点的服务端算子插件（随节点包自注册）。

v1.3.0 起改为族版本化注册（F1/dev-plan §47）：注册名带 @<内容hash12> 后缀、
落盘名 controlnet_apply_<hash>.py，与旧版 sd_controlnet_apply.py 并存互不顶替。

sd.controlnet.apply_masked：在 sd.controlnet.apply 基础上新增可选 mask 输入
（IMAGE 灰度，白=重绘区）——给出时把 hint 图蒙版区烧黑（置 0）后再捆绑，
即 lllyasviel control_v11p_sd15_inpaint 官方 hint 约定的 [0,1] 近似
（官方为 [-1,1] 值域烧 -1；PIL 只能表达 [0,255]，黑=0，ComfyUI 同款近似）。
mask 为空时行为与 sd.controlnet.apply 完全一致。
"""
from app import ops

# --- F1 族版本化：算子身份 = 实现内容 --------------------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
OP_VERSION = "05eab95d60a0"
OP_FAMILY = "controlnet_apply"  # 落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs,
                                    outputs, description)
# ---------------------------------------------------------------------------


def _burn_mask(image, mask):
    """hint 图蒙版区烧黑：mask（L 灰度，白=重绘区）>127 的像素置 (0,0,0)。"""
    import numpy as np
    img = image.convert("RGB")
    m = mask.convert("L")
    if m.size != img.size:
        m = m.resize(img.size)
    arr = np.asarray(img).copy()
    black = np.asarray(m) > 127
    arr[black] = 0
    from PIL import Image
    out = Image.fromarray(arr, "RGB")
    print(f"[controlnet.apply_masked] hint 烧黑：蒙版像素 {int(black.sum())} "
          f"/ {black.size}（{black.sum() / black.size:.1%}）", flush=True)
    return out


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "sd.controlnet.apply_masked",
        inputs={"control_net": "CONTROL_NET", "image": "IMAGE",
                "mask": "IMAGE",
                "strength": "FLOAT", "start_percent": "FLOAT",
                "end_percent": "FLOAT", "control": "CONTROL"},
        outputs={"control": "CONTROL"},
        description="ControlNet Apply（带蒙版烧黑）：mask 非空时把 hint 图蒙版区"
                    "（白=重绘区）烧黑后捆绑——control_v11p_sd15_inpaint 的 hint"
                    " 约定；mask 为空等价 sd.controlnet.apply。control 可串接上一个"
                    " apply 的 CONTROL 组成多 CN 链（残差×各自 strength 求和）")
    def _op_controlnet_apply_masked(control_net, image, mask=None, strength=1.0,
                                    start_percent=0.0, end_percent=1.0,
                                    control=None):
        if mask is not None:
            image = _burn_mask(image, mask)
        return ops.controlnet_apply(control_net, image, strength,
                                    start_percent, end_percent, control=control)
