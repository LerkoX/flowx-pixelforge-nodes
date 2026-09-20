"""sd3-clip-encode 节点的服务端算子插件（随节点包自注册）。

SD3.5 文本编码（对标 ComfyUI CLIP Text Encode）：双 CLIP-L/G 编码 + pooled 投影，
T5 弃用时管道自动零填充。不手写编码数学——直接调管道的 encode_prompt，
与 diffusers 行为逐位一致（含 zero-pad 到 joint_attention_dim=4096）。

COND 对象 = {"embeds": (b,154,4096) 张量, "pooled": (b,2048) 张量}；
正向/反向各用一个节点实例（与 SD1.x 的 clip.encode 同用法）。
"""
import torch

from app import ops

SD3_CLASS = "StableDiffusion3Pipeline"


def register(registry):
    @registry.register(
        "sd3.clip.encode",
        inputs={"clip": "CLIP", "text": "STRING"},
        outputs={"cond": "COND", "info": "STRING"},
        description="SD3.5 CLIP Text Encode：文本 → conditioning（双 CLIP-L/G + pooled；"
                    "T5 弃用时自动零填充）。正向/反向各用一个节点实例；"
                    "info 输出 embed 形状/范数统计供画布观察")
    def _op_sd3_clip_encode(clip, text=""):
        pipe, _ = ops.resolve_pipe(clip)
        cls_name = type(pipe).__name__
        if cls_name != SD3_CLASS:
            raise ValueError(
                f"sd3.clip.encode 只接受 {SD3_CLASS} 的 CLIP 视图（当前 {cls_name}）")
        device = ops.exec_device_of(pipe)
        with torch.no_grad():
            embeds, _neg, pooled, _neg_pooled = pipe.encode_prompt(
                prompt=text, prompt_2=None, prompt_3=None,
                do_classifier_free_guidance=False, device=device)
        cond = {"embeds": embeds, "pooled": pooled}
        info = (f"embeds={tuple(embeds.shape)} norm={embeds.float().norm():.2f} | "
                f"pooled={tuple(pooled.shape)} norm={pooled.float().norm():.2f} | "
                f"len={len(text)}")
        print(f"[sd3.clip.encode] {info}", flush=True)
        return {"cond": cond, "info": info}
