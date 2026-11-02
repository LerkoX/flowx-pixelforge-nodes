"""image-sharpen：锐化（对标 ComfyUI），纯节点侧 PIL。

v1.1.0：可选 mask 输入——只在蒙版白区应用锐化，蒙版外逐像素保留原图。
局部重绘流水线里蒙版外的像素本来就是原图，整图锐化会把原图细节（衣物边缘、
蕾丝等）锐化成扎眼的细线，故提供蒙版限定能力。
"""
from flowx_client import emit_input_media, emit, emit_result_preview, param, token
from image_filters import sharpen
from image_io import get_pimg, post_pimg


def process(im, mode="usm", radius=2.0, percent=150.0, threshold=3.0, mask=None):
    out = sharpen(im, mode=mode, radius=radius, percent=percent,
                  threshold=threshold)
    if mask is None:
        return out
    m = mask.convert("L")
    if m.size != im.size:
        m = m.resize(im.size)
    base = im.convert("RGB").copy()
    base.paste(out.convert("RGB"), (0, 0), m)
    return base


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    # 输入原图物化：画布对比控件的「原图」侧（推理服务对象 GC/TTL 后
    # 必然 404，必须在执行当时落一份到 Studio 媒体缓存）
    emit_input_media(url, param("image"), tok)
    im = get_pimg(url, param("image"), tok, mode="RGB")
    mask_id = param("mask", "")
    mask = get_pimg(url, mask_id, tok, mode="L") if mask_id else None
    out = process(im, mode=param("mode", "usm"),
                  radius=param("radius", 2.0, cast=float),
                  percent=param("percent", 150.0, cast=float),
                  threshold=param("threshold", 3.0, cast=float),
                  mask=mask)
    oid = post_pimg(url, out, tok)
    print(f"[image-sharpen] image={param('image')} mask={mask_id or '-'} -> image={oid}",
          flush=True)
    emit(image=oid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
