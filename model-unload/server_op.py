"""model-unload 节点的服务端算子插件（随节点包自注册）。

model.unload 算子与 POST /model/unload 端点同语义（端点保留供运维 curl 直调，
算子形态供图编排/inference-op 使用），委托 app.ops.model_unload。
"""
from app import ops


def register(registry):
    @registry.register(
        "model.unload",
        inputs={"target": "STRING"},
        outputs={"unloaded": "STRING", "resident": "STRING"},
        description="卸载常驻模型腾显存：target 空/all/*=全部，否则按名字或缓存键匹配"
                    "（可省略扩展名，支持 vae:/cn:/motion 组合管）。走与加载一致的淘汰路径"
                    "（断对象仓库视图 → gc → empty_cache）；输出 unloaded/resident 逗号分隔清单。"
                    "典型用法：视频/大模型段跑完先卸载，再跑图像采样")
    def _op_model_unload(target=""):
        from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化
        return ops.model_unload(models, target)
