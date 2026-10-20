"""flux-lora-load 节点的服务端算子插件（随节点包自注册，不动服务端核心）。

给 FLUX GGUF transformer（FluxTransformer2DModel）挂 LoRA 增量补丁：
- 复用 diffusers FluxLoraLoaderMixin 的键名转换与 PEFT 注入——用轻量
  FluxPipeline 壳（scheduler/vae/text_encoder 全 None，仅借 transformer
  引用）调用 load_lora_weights + set_adapters，壳随即销毁，补丁留在
  transformer 本体上（GGUF 量化层是 nn.Linear 子类，PEFT 正常注入，
  delta 以 compute dtype 单独计算，无需反量化底权重）
- LoRA 文件定位复用 model_manager.resolve_lora（LORAS_DIR，可省略扩展名）
- 幂等：同名 adapter 重复挂载先 delete_adapters 再重载（换强度/换文件同路径）
- 注意：补丁打在 LRU 常驻的共享 transformer 对象上，属进程级状态；
  strength=0 等价于停用但不移除。多流水线串用同一底模时后挂的 LoRA 会
  影响先挂的（GPU 盒单用户 FIFO 场景可接受，如需纯净底模重新 unet_load
  无效——对象是同一份，须 unload 后重载）

算子：
- flux.lora_apply(model, lora, strength) → MODEL（同一 transformer 引用）
"""
import os
import time

from app import model_manager as _mm  # noqa: F401  （LORAS_DIR 常量来源）


def register(registry):
    @registry.register(
        "flux.lora_apply",
        inputs={"model": "MODEL", "lora": "STRING", "strength": "FLOAT"},
        outputs={"model": "MODEL", "info": "STRING"},
        description="FLUX LoRA 挂载：给 flux.unet_load 的 MODEL（GGUF "
                    "transformer）挂 LoRA 增量补丁（PEFT，可多次叠加不同 "
                    "adapter；同名重挂先卸再载）。lora 为 LORAS_DIR 下文件名"
                    "（可省略扩展名），strength 默认 1.0、0=停用")
    def _op_flux_lora_apply(model, lora, strength=1.0):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化
        from diffusers import FluxPipeline

        if type(model).__name__ != "FluxTransformer2DModel":
            raise ValueError(
                "flux.lora_apply: model 需为 flux.unet_load 的输出 "
                f"（当前 {type(model).__name__}）")

        path, lora_name = models.resolve_lora(lora)
        adapter = lora_name.replace(" ", "_")
        strength = float(strength)
        t0 = time.time()

        # 同名 adapter 已挂：先卸再载（换强度/同路径换文件都走这里，幂等）
        applied = getattr(model, "_flux_lora_adapters", None)
        if applied is None:
            applied = {}
            model._flux_lora_adapters = applied
        if adapter in applied:
            try:
                model.delete_adapters(adapter)
            except Exception as e:
                print(f"[flux.lora_apply] delete_adapters({adapter}) warn: {e}",
                      flush=True)
            applied.pop(adapter, None)

        # 轻量 FluxPipeline 壳：仅借用 transformer 引用，复用 diffusers 的
        # LoRA 键名转换/PEFT 注入；壳销毁不影响 transformer 本体上的补丁
        pipe = FluxPipeline(scheduler=None, vae=None, text_encoder=None,
                            tokenizer=None, text_encoder_2=None,
                            tokenizer_2=None, transformer=model)
        try:
            pipe.load_lora_weights(os.path.dirname(path) or ".",
                                   weight_name=os.path.basename(path),
                                   adapter_name=adapter)
            pipe.set_adapters([adapter], [strength])
        finally:
            pipe.transformer = None
            del pipe

        applied[adapter] = strength
        info = (f"adapter={adapter} strength={strength} "
                f"file={os.path.basename(path)} "
                f"({os.path.getsize(path)/1e6:.1f}MB) "
                f"{time.time()-t0:.1f}s total_adapters={len(applied)}")
        print(f"[flux.lora_apply] {info}", flush=True)
        return {"model": model, "info": info}
