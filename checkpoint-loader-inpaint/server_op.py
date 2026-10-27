"""checkpoint-loader-inpaint 节点的服务端算子插件。

SD1.x Inpainting checkpoint（9 通道 UNet）加载：现有 sniff 对 .ckpt 只能
按普通 SD1.5 处理，这里在一次 checkpoint_load 调用期间把文件名包含
inpaint/inpainting 的单文件临时改写为 StableDiffusionInpaintPipeline。
"""
import os

from app import ops, sniff


_INPAINT_CLS = "StableDiffusionInpaintPipeline"


def register(registry):
    @registry.register(
        "sd.checkpoint.load_inpaint",
        inputs={"ckpt": "STRING", "dtype": "STRING", "offload": "STRING",
                "use_t5": "STRING"},
        outputs={"model": "MODEL", "clip": "CLIP", "vae": "VAE"},
        description="加载 SD1.x Inpainting checkpoint（9 通道 UNet），输出 "
                    "model/clip/vae 三视图；文件名需包含 inpaint/inpainting，"
                    "其余语义与 sd.checkpoint.load 一致")
    def _op_checkpoint_load_inpaint(ckpt, dtype="auto", offload="auto",
                                    use_t5="auto"):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化

        orig = sniff.sniff_arch

        def patched(path):
            loader, cls_name = orig(path)
            base = os.path.basename(path).lower()
            if loader == "single_file" and cls_name == "StableDiffusionPipeline" \
                    and ("inpaint" in base or "inpainting" in base):
                print(f"[checkpoint-load-inpaint] {base} -> {_INPAINT_CLS}",
                      flush=True)
                return loader, _INPAINT_CLS
            return loader, cls_name

        sniff.sniff_arch = patched
        try:
            return ops.checkpoint_load(models, ckpt, dtype, offload, use_t5)
        finally:
            sniff.sniff_arch = orig
