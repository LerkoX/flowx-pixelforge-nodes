"""lora-loader 节点的服务端算子插件（随节点包自注册）。

SD1.x/SDXL LoRA 挂载：委托 app.ops.lora_apply（ModelPatcher 补丁视图，可串联叠加）。
"""
from app import ops


def register(registry):
    @registry.register(
        "sd.lora.apply",
        inputs={"model": "MODEL", "lora": "STRING", "strength": "FLOAT"},
        outputs={"model": "MODEL", "clip": "CLIP"},
        description="LoRA 加载：给 MODEL 挂增量补丁（可多个串联叠加），strength 默认 1.0；"
                    "lora 为 LORAS_DIR 下文件名（可省略扩展名）")
    def _op_lora_apply(model, lora, strength=1.0):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化
        return ops.lora_apply(models, model, lora, strength)
