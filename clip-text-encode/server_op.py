"""clip-text-encode 节点的服务端算子插件（随节点包自注册）。

SD1.x/SDXL CLIP 文本编码：委托 app.ops.clip_encode（按底模架构分派，
SDXL 双编码器 + pooled，采样时自动注入 added_cond_kwargs）。
"""
from app import ops


def register(registry):
    @registry.register(
        "sd.clip.encode",
        inputs={"clip": "CLIP", "text": "STRING", "width": "INT",
                "height": "INT", "clip_skip": "INT"},
        outputs={"cond": "COND"},
        description="CLIP Text Encode：文本 → conditioning。按底模架构分派（M5）："
                    "SD1.x 单 text_encoder → 张量 COND；SDXL 双 text encoder → "
                    "SDXLCond（embeds + pooled），采样时自动注入 added_cond_kwargs"
                    "（text_embeds/time_ids）。width/height 仅 SDXL 有意义："
                    "原始尺寸（0=auto，由采样 latent 推导，影响 time_ids）；"
                    "clip_skip 负值跳过末 n 层（SDXL 常用 -2），0=默认")
    def _op_clip_encode(clip, text, width=0, height=0, clip_skip=0):
        return ops.clip_encode(clip, text, width, height, clip_skip)
