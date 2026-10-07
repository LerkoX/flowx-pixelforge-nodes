"""vae-load 节点的服务端算子插件（随节点包自注册）。

外挂 VAE 加载（vae-ft-mse 等提升解码质量）：走 model_manager LRU 常驻。
"""
def register(registry):
    @registry.register(
        "sd.vae.load",
        inputs={"name": "STRING", "dtype": "STRING"},
        outputs={"vae": "VAE"},
        description="Load VAE：单独加载 VAE 组件（MODELS_DIR 下 safetensors 单文件或 "
                    "diffusers 组件目录），输出可直接喂 sd.vae.decode/sd.vae.encode 替代管道内置 "
                    "VAE（外挂 vae-ft-mse 等提升解码质量）。dtype=auto/fp16/fp32，auto "
                    "默认 fp32（外挂 VAE 的意义即解码质量）；进同一 LRU 常驻管理")
    def _op_vae_load(name, dtype="auto"):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化
        key, _newly = models.load_vae(name, dtype)
        return {"vae": models.get(key)}
