"""embedding-load 节点的服务端算子插件（随节点包自注册）。

SD1.x/SDXL Textual Inversion 加载：EMBEDDINGS_DIR 下的 embedding 载进 CLIP
文本编码器，委托 app.ops.embedding_load（幂等，已加载自动跳过）。
"""
import os

from app import ops

EMBEDDINGS_DIR = os.environ.get("EMBEDDINGS_DIR", "/models/embeddings")


def register(registry):
    @registry.register(
        "sd.embedding.load",
        inputs={"clip": "CLIP", "names": "STRING"},
        outputs={"clip": "CLIP"},
        description="Textual Inversion 加载：把 EMBEDDINGS_DIR 下的 embedding（逗号分隔，"
                    "如 badhandv4,EasyNegative）载进 CLIP 文本编码器；之后正/反提示词里直接写"
                    "该词生效（常用于负面压制缺陷）。幂等，已加载自动跳过")
    def _op_embedding_load(clip, names):
        return ops.embedding_load(clip, EMBEDDINGS_DIR, names)
