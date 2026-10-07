"""checkpoint-loader 节点的服务端算子插件（随节点包自注册，经 POST /admin/plugins 上传）。

SD1.x/SDXL checkpoint 加载原语：委托 app.ops.checkpoint_load（diffusers
from_single_file/from_pretrained 双路径），输出 model/clip/vae 三对象。
"""
from app import ops


def register(registry):
    @registry.register(
        "sd.checkpoint.load",
        inputs={"ckpt": "STRING", "dtype": "STRING", "offload": "STRING",
                "use_t5": "STRING"},
        outputs={"model": "MODEL", "clip": "CLIP", "vae": "VAE"},
        description="加载 checkpoint 到显存（幂等），输出 model/clip/vae 三个对象。"
                    "性能旋钮（dev-plan 9.6 纪律 #3）：dtype=auto/fp16/bf16/fp32、"
                    "offload=auto/none/model/sequential、use_t5=auto/on/off（仅 SD3 生效，"
                    "auto 继承 SD3_USE_T5 环境变量，默认弃用 T5-XXL）；auto 继承服务端环境变量，"
                    "不同旋钮组合是独立常驻条目（同一 LRU 管理）")
    def _op_checkpoint_load(ckpt, dtype="auto", offload="auto", use_t5="auto"):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化
        return ops.checkpoint_load(models, ckpt, dtype, offload, use_t5)
