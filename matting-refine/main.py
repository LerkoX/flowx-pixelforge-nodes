"""matting-refine：抠图 alpha 精修（纯节点侧 PIL，不占显存）。

一次算子做完：二值化阈值 → 伽马/对比度曲线 → 扩缩 → 羽化 → 去小连通域 → 反相。
单一操作请用 mask-grow / mask-feather / mask-invert（避免重复投入）。
"""
from flowx_client import emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg, pbool


def _lut(contrast=1.0, gamma=1.0, threshold=0):
    """把 对比度 → 伽马 → 阈值 三步合成一张 256 项 LUT（纯 PIL，无 numpy）。"""
    lut = []
    for i in range(256):
        v = float(i)
        if contrast != 1.0:
            v = (v - 127.5) * contrast + 127.5
        if gamma != 1.0:
            v = 255.0 * (max(0.0, min(255.0, v)) / 255.0) ** (1.0 / max(1e-3, gamma))
        v = max(0.0, min(255.0, v))
        if threshold > 0:
            v = 255.0 if v >= threshold else 0.0
        lut.append(int(round(v)))
    return lut


def process(mask, threshold=0, gamma=1.0, contrast=1.0, feather=0, grow=0,
            despeckle=0, invert=False):
    """灰度 alpha 精修（纯 PIL，不依赖 numpy）。

    顺序：对比度/伽马/阈值（合并成一张 LUT）→ 扩缩 → 去杂点 → 羽化 → 反相。
    单一操作请用 mask-grow / mask-feather / mask-invert；本节点用于串联调参。
    """
    from PIL import Image, ImageFilter, ImageOps

    m = mask.convert("L")

    # 1) 曲线 + 二值化（一张 LUT 搞定；阈值 0 = 不二值化，保留灰阶羽化）
    if float(contrast) != 1.0 or float(gamma) != 1.0 or int(threshold) > 0:
        m = m.point(_lut(float(contrast), float(gamma), int(threshold)))

    # 2) 扩缩（MaxFilter = 扩张，MinFilter = 收缩，逐像素 3×3）
    g = int(grow)
    for _ in range(abs(g)):
        m = m.filter(ImageFilter.MaxFilter(3) if g > 0 else ImageFilter.MinFilter(3))

    # 3) 去杂点（形态学开运算：先腐蚀后膨胀 ⇒ 抹掉尺度 < k/2 的孤立点，
    #    主体边界被腐掉的部分随后被膨胀还原；k 为奇数）
    k = int(despeckle)
    if k >= 3:
        k = k if k % 2 == 1 else k + 1
        m = m.filter(ImageFilter.MinFilter(k)).filter(ImageFilter.MaxFilter(k))

    # 4) 羽化
    if float(feather) > 0:
        m = m.filter(ImageFilter.GaussianBlur(float(feather)))

    # 5) 反相
    if invert:
        m = ImageOps.invert(m)
    return m


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    m = get_pimg(url, param("mask"), tok, mode="L")
    out = process(m, threshold=param("threshold", 0, cast=int),
                  gamma=param("gamma", 1.0, cast=float),
                  contrast=param("contrast", 1.0, cast=float),
                  feather=param("feather", 0, cast=float),
                  grow=param("grow", 0, cast=int),
                  despeckle=param("despeckle", 0, cast=int),
                  invert=pbool(param("invert", "false")))
    oid = post_pimg(url, out, tok)
    print(f"[matting-refine] mask={param('mask')} -> mask={oid}", flush=True)
    emit(mask=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
