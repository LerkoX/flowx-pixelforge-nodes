"""load-image 节点的服务端算子插件（随节点包自注册）。

image.load：读 INPUT_DIR 下的服务端本地图片（仅文件名）。客户端本地图
走 POST /images 上传（本节点 main.py 的路径），本算子保留服务端本地图能力。
"""
import os

from app import ops

INPUT_DIR = os.environ.get("INPUT_DIR", "/input")


def register(registry):
    @registry.register(
        "image.load",
        inputs={"name": "STRING"},
        outputs={"image": "IMAGE"},
        description="Load Image：读 INPUT_DIR 下的服务端本地图片（仅文件名）；客户端上传用 POST /images")
    def _op_image_load(name):
        return ops.image_load(INPUT_DIR, name)
