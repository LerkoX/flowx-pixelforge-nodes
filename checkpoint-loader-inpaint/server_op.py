"""checkpoint-loader-inpaint 节点的服务端算子插件。

Inpainting checkpoint（9 通道 UNet）加载：sniff 无法从内容判断"这是 inpaint
权重"，这里在一次 checkpoint_load 调用期间按路径名（文件名或 diffusers 目录名）
含 inpaint/inpainting 时强制映射到 *InpaintPipeline：

    StableDiffusionPipeline   -> StableDiffusionInpaintPipeline
    StableDiffusionXLPipeline -> StableDiffusionXLInpaintPipeline

映射必要性：非 Inpaint 管道没有 `mask_processor`，`sd.vae.encode_inpaint_masked`
与 diffusers 原生 inpaint 语义都会失败。
"""
import os

from app import ops, sniff


_MAP = {
    "StableDiffusionPipeline": "StableDiffusionInpaintPipeline",
    "StableDiffusionXLPipeline": "StableDiffusionXLInpaintPipeline",
}


def register(registry):
    @registry.register(
        "sd.checkpoint.load_inpaint",
        inputs={"ckpt": "STRING", "dtype": "STRING", "offload": "STRING",
                "use_t5": "STRING"},
        outputs={"model": "MODEL", "clip": "CLIP", "vae": "VAE"},
        description="加载 Inpainting checkpoint（9 通道 UNet，SD1.x/SDXL），输出 "
                    "model/clip/vae 三视图；路径（文件名或 diffusers 目录名）需包含 "
                    "inpaint/inpainting，"
                    "其余语义与 sd.checkpoint.load 一致")
    def _op_checkpoint_load_inpaint(ckpt, dtype="auto", offload="auto",
                                    use_t5="auto"):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化

        orig = sniff.sniff_arch

        def patched(path):
            loader, cls_name = orig(path)
            base = os.path.basename(str(path).rstrip("/")).lower()
            if "inpaint" in base and cls_name in _MAP:
                print(f"[checkpoint-load-inpaint] {base} : {cls_name} -> "
                      f"{_MAP[cls_name]}", flush=True)
                return loader, _MAP[cls_name]
            return loader, cls_name

        sniff.sniff_arch = patched
        try:
            return ops.checkpoint_load(models, ckpt, dtype, offload, use_t5)
        finally:
            sniff.sniff_arch = orig
