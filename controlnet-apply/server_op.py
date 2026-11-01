"""controlnet-apply 节点的服务端算子插件（随节点包自注册）。

族版本化注册（F1/dev-plan §47）：注册名带 @<内容hash12> 后缀、
落盘名 controlnet_apply_<hash>.py，与旧版算子并存互不顶替。

sd.controlnet.apply_masked：在 sd.controlnet.apply 基础上新增可选 mask 输入
（IMAGE 灰度，白=重绘区）。

v1.3.1 修正 inpaint hint 编码：官方约定（lllyasviel make_inpaint_condition）
蒙版区像素 = -1.0（"待重画"魔法值），而非黑色 0——填 0 会被模型当作真实黑色
内容保留（实测 exec 1064：整个人形被引导成纯黑）。PIL 无法表达负值，故构造
_MaskedHint 鸭子类型（numpy float 数组，蒙版区 = -255；core sample 的
`np.asarray(img.convert("RGB"))/255.0` 后正好 -1.0），直接组装
ops.ControlBundle 绕过 core 的 PIL isinstance 校验。mask 为空时委托
core controlnet_apply，行为与 sd.controlnet.apply 完全一致。
"""
from app import ops

# --- F1 族版本化：算子身份 = 实现内容 --------------------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
OP_VERSION = "48a1084ed588"
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


class _MaskedHint:
    """鸭子类型 PIL 替代：float32 HxWx3，蒙版区 = -255（/255 后 = -1.0，
    即官方 inpaint CN hint 约定）。

    core sample 对 bundle.image 的全部消费仅三个调用：
      img.size / img.resize((wp, hp), Image.BILINEAR) / np.asarray(img.convert("RGB"), ...)
    本类逐一实现：resize 时图像区与蒙版区分别双线性缩放后再合成（边界半烧黑
    只影响 1-2px 过渡带，可忽略）。
    """

    def __init__(self, arr):
        self._arr = arr

    @property
    def size(self):
        h, w = self._arr.shape[:2]
        return (w, h)

    def resize(self, size, resample=None):
        import numpy as np
        from PIL import Image
        img_part = Image.fromarray(
            np.clip(self._arr, 0, 255).astype("uint8"), "RGB"
        ).resize(size, Image.BILINEAR)
        m = Image.fromarray(
            (self._arr.min(axis=2) < 0).astype("uint8") * 255, "L"
        ).resize(size, Image.BILINEAR)
        a = np.asarray(img_part, dtype=np.float32)
        a[np.asarray(m) > 127] = -255.0
        return _MaskedHint(a)

    def convert(self, mode):
        # np.asarray(ndarray, dtype=np.float32) 直通；/255 后蒙版区 = -1.0
        return self._arr


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "sd.controlnet.apply_masked",
        inputs={"control_net": "CONTROL_NET", "image": "IMAGE",
                "mask": "IMAGE",
                "strength": "FLOAT", "start_percent": "FLOAT",
                "end_percent": "FLOAT", "control": "CONTROL"},
        outputs={"control": "CONTROL"},
        description="ControlNet Apply（inpaint hint 版）：mask 非空时 hint 蒙版区"
                    "（白=重绘区）填 -1.0（官方 make_inpaint_condition 约定，"
                    "control_v11p_sd15_inpaint 必需，填 0 会被当成黑色内容）；"
                    "mask 为空等价 sd.controlnet.apply。control 可串接上一个 apply"
                    " 的 CONTROL 组成多 CN 链（残差×各自 strength 求和）")
    def _op_controlnet_apply_masked(control_net, image, mask=None, strength=1.0,
                                    start_percent=0.0, end_percent=1.0,
                                    control=None):
        if mask is None:
            return ops.controlnet_apply(control_net, image, strength,
                                        start_percent, end_percent,
                                        control=control)
        import numpy as np
        # 复刻 core controlnet_apply 的入参校验（步窗口部分）
        if not 0.0 <= float(start_percent) < 1.0:
            raise ValueError(f"start_percent 需在 [0,1)，got {start_percent}")
        if not 0.0 < float(end_percent) <= 1.0:
            raise ValueError(f"end_percent 需在 (0,1]，got {end_percent}")
        if float(end_percent) <= float(start_percent):
            raise ValueError(f"步窗口为空：[{start_percent},{end_percent})")
        img = image.convert("RGB")
        m = mask.convert("L")
        if m.size != img.size:
            m = m.resize(img.size)
        arr = np.asarray(img, dtype=np.float32).copy()
        burnt = np.asarray(m) > 127
        arr[burnt] = -255.0
        print(f"[controlnet.apply_masked] hint 蒙版区 = -1.0："
              f"{int(burnt.sum())} / {burnt.size}（{burnt.sum() / burnt.size:.1%}）",
              flush=True)
        bundle = ops.ControlBundle(control_net, _MaskedHint(arr),
                                   float(strength), float(start_percent),
                                   float(end_percent))
        if control is None:
            print(f"[controlnet.apply_masked] strength={strength} "
                  f"window=[{start_percent},{end_percent})", flush=True)
            return {"control": bundle}
        chain = list(control) if isinstance(control, list) else [control]
        chain.append(bundle)
        print(f"[controlnet.apply_masked] strength={strength} "
              f"window=[{start_percent},{end_percent}) "
              f"（串接为 {len(chain)} 路 ControlNet）", flush=True)
        return {"control": chain}
