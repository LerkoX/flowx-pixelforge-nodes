"""catvton-tryon：CatVTON 真换装（vton.catvton_tryon，服务端插件）。

person（模特图）+ cloth（衣服参考图）+ mask（白=要换掉的旧衣服区域）→ 换装结果图。
mask 语义与 mask-paint / mask-grow 一致（255=重绘区），可直接复用现有手绘笔画；
`repaint=true` 时按原图坐标系贴回：蒙版外逐像素保留原图（脸/背景零改动），
蒙版外沿用高斯模糊过渡（mask_blur）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("vton.catvton_tryon", {
        "vton": param("vton"),
        "person": ref(param("person")),
        "cloth": ref(param("cloth")),
        "mask": ref(param("mask")),
        "steps": int(param("steps", "50")),
        "guidance": float(param("guidance", "2.5")),
        "seed": int(param("seed", "-1")),
        "width": int(param("width", "768")),
        "height": int(param("height", "1024")),
        "mask_blur": int(param("mask_blur", "9")),
        "repaint": param("repaint", "true") == "true",
        "preview_every": int(param("preview_every", "5")),
    }, emit_keys=["image", "info"], timeout=3600)


if __name__ == "__main__":
    main()
