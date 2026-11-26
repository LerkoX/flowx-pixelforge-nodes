"""image-upscale：图片放大（纯节点侧 PIL，对标 ComfyUI ImageUpscale）。

历史：旧版走服务端 ops.image_upscale（不存在该算子 + 裸名 403 守卫双杀）。
PIL resize 足以覆盖角色卡裁剪放大的用途，改为纯客户端（对齐 image-crop）。
"""
from flowx_client import emit_input_media, emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg


def _resample(name):
    from PIL import Image as _I
    table = {"lanczos": _I.LANCZOS, "bicubic": _I.BICUBIC, "bilinear": _I.BILINEAR,
             "nearest": _I.NEAREST, "area": _I.BOX}
    return table.get(str(name or "lanczos").lower(), _I.LANCZOS)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    emit_input_media(url, param("image"), tok)
    im = get_pimg(url, param("image"), tok, mode="RGB")

    w, h = param("width", 0, cast=int), param("height", 0, cast=int)
    scale = param("scale", 2.0, cast=float)
    keep_aspect = str(param("keep_aspect", True)).lower() in ("1", "true", "yes")

    if w > 0 and h > 0:
        target = (w, h)
        if keep_aspect:
            im2 = im.copy()
            im2.thumbnail(target, _resample(param("algorithm", "lanczos")))
            target = im2.size
            im = im2
        else:
            im = im.resize(target, _resample(param("algorithm", "lanczos")))
    else:
        target = (max(1, round(im.width * scale)), max(1, round(im.height * scale)))
        im = im.resize(target, _resample(param("algorithm", "lanczos")))

    oid = post_pimg(url, im, tok)
    print(f"[image-upscale] {param('image')} {target[0]}x{target[1]} -> {oid}", flush=True)
    emit(image=oid, width=im.width, height=im.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
