"""preprocess-canny 节点的服务端算子插件（随节点包自注册，经 POST /admin/plugins 上传）。

从输入图像提取 canny 风格线稿（PIL FIND_EDGES 轻量实现，与 accept-tier3
验收脚本的 hint 生成逻辑一致），输出黑底白线线稿图，供 ControlNet
（control_v11p_sd15_canny）作 hint 控制构图。
"""
from PIL import ImageFilter, ImageOps


def register(registry):
    @registry.register(
        "preprocess.canny",
        inputs={"image": "IMAGE", "passes": "INT", "cutoff": "INT"},
        outputs={"image": "IMAGE"},
        description="Canny 风格线稿提取：灰度化 → FIND_EDGES × passes → "
                    "autocontrast(cutoff)，输出黑底白线 RGB 线稿图，"
                    "供 controlnet.apply 的 image（hint）输入使用；"
                    "batch 列表输入逐张处理")
    def _op_preprocess_canny(image, passes=2, cutoff=1):
        def _edge(im):
            g = im.convert("L")
            for _ in range(max(1, int(passes))):
                g = g.filter(ImageFilter.FIND_EDGES)
            g = ImageOps.autocontrast(g, cutoff=max(0, int(cutoff)))
            return g.convert("RGB")

        if isinstance(image, list):
            return {"image": [_edge(i) for i in image]}
        return {"image": _edge(image)}
