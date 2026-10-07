"""controlnet-load 节点的服务端算子插件（随节点包自注册）。

SD1.x/SDXL ControlNet 组件加载：走 model_manager LRU 常驻。
"""
def register(registry):
    @registry.register(
        "sd.controlnet.load",
        inputs={"name": "STRING", "dtype": "STRING"},
        outputs={"control_net": "CONTROL_NET"},
        description="ControlNet Loader：单独加载 ControlNet 组件（MODELS_DIR 下 "
                    "control_v11* 等 safetensors 单文件或 diffusers 组件目录），"
                    "dtype=auto/fp16/bf16/fp32，auto 默认 fp16；进同一 LRU 常驻管理。"
                    "输出经 sd.controlnet.apply 捆绑 hint 图后喂 sd.sample 的 control 端口")
    def _op_controlnet_load(name, dtype="auto"):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化
        key, _newly = models.load_controlnet(name, dtype)
        return {"control_net": models.get(key)}
