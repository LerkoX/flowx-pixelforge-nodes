"""motion-loader 节点的服务端算子插件（随节点包自注册）。

AnimateDiff 组合加载：SD1.x checkpoint + MotionAdapter → AnimateDiffPipeline，
委托 app.ops.motion_load。
"""
from app import ops


def register(registry):
    @registry.register(
        "sd.motion.load",
        inputs={"ckpt": "STRING", "motion": "STRING"},
        outputs={"model": "MODEL", "clip": "CLIP", "vae": "VAE"},
        description="Motion 加载：SD1.x checkpoint + MotionAdapter → AnimateDiffPipeline（文生视频）。"
                    "ckpt 为 MODELS_DIR 下底模名（同 sd.checkpoint.load）；motion 为 MODELS_DIR/motion/ 下的"
                    " diffusers 目录名或 safetensors 文件名（可省略扩展名）。直接收底模名而非 MODEL 引用："
                    "组合需全新实例化底模，预先 sd.checkpoint.load 会白占一份内存")
    def _op_motion_load(ckpt, motion):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化
        return ops.motion_load(models, ckpt, motion)
