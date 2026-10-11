"""image-pad：四边扩画布 + 扩图蒙版（对标 ComfyUI Pad Image for Outpainting / ImagePadForOutpaint）。

纯节点侧 PIL。输出两张图，配合既有 inpaint 链即可做 outpainting：
    image-pad → vae-encode → latent-set-noise-mask → ksampler(denoise=1.0) → vae-decode → save-image

- `image`：扩边后的图。fill=edge（默认）用边缘像素拉伸补边（比纯色更利于扩散模型延续构图）；
  black/white/gray 为纯色填充。
- `mask`：扩图蒙版（灰度 IMAGE 对象）。原图区域 0（保留）、新扩区域 255（重绘），
  feather>0 时边界按该宽度羽化过渡；全部扩边为 0 时输出全 0 蒙版（下游等价于不做改动）。

⚠️ inpaint 必须用 denoise=1.0（§三十九判决实验：denoise<1 从加噪原图出发会几乎不改变内容）。
"""
from PIL import Image, ImageFilter

from flowx_client import emit, emit_result_preview, param, token
from image_io import get_pimg, post_pimg

FILLS = {"black": (0, 0, 0), "white": (255, 255, 255), "gray": (128, 128, 128)}


def _stretch(im, box, size):
    """取 box 区域拉伸到 size（NEAREST：边缘像素复制语义，避免插值糊边）。"""
    return im.crop(box).resize(size, Image.Resampling.NEAREST)


def _edge_expand(im, left, right, top, bottom):
    """用边缘像素拉伸补边（ImageOps.expand 只能纯色填充）。"""
    w, h = im.size
    W, H = w + left + right, h + top + bottom
    canvas = Image.new(im.mode, (W, H))
    canvas.paste(im, (left, top))
    if top:
        canvas.paste(_stretch(im, (0, 0, w, 1), (w, top)), (left, 0))
    if bottom:
        canvas.paste(_stretch(im, (0, h - 1, w, h), (w, bottom)), (left, top + h))
    if left:
        canvas.paste(_stretch(im, (0, 0, 1, h), (left, h)), (0, top))
        if top:
            canvas.paste(_stretch(im, (0, 0, 1, 1), (left, top)), (0, 0))
        if bottom:
            canvas.paste(_stretch(im, (0, h - 1, 1, h), (left, bottom)), (0, top + h))
    if right:
        canvas.paste(_stretch(im, (w - 1, 0, w, h), (right, h)), (left + w, top))
        if top:
            canvas.paste(_stretch(im, (w - 1, 0, w, 1), (right, top)), (left + w, 0))
        if bottom:
            canvas.paste(_stretch(im, (w - 1, h - 1, w, h), (right, bottom)),
                         (left + w, top + h))
    return canvas


def build_mask(size, inner, feather):
    """扩图蒙版：inner=(x,y,w,h) 为原图区域（0），其余 255；feather>0 时边界羽化。"""
    W, H = size
    mask = Image.new("L", (W, H), 255)
    mask.paste(Image.new("L", (inner[2], inner[3]), 0), (inner[0], inner[1]))
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(float(feather) / 2.0))
    return mask


def process(im, left=0, right=0, top=0, bottom=0, feather=32, fill="edge"):
    left, right, top, bottom = int(left), int(right), int(top), int(bottom)
    if min(left, right, top, bottom) < 0:
        raise ValueError("pad values must be >= 0")
    w, h = im.size
    if left == right == top == bottom == 0:
        return im.convert("RGB"), build_mask((w, h), (0, 0, w, h), 0)
    mode = str(fill or "edge").strip().lower()
    rgb = im.convert("RGB")
    if mode == "edge":
        out = _edge_expand(rgb, left, right, top, bottom)
    elif mode in FILLS:
        canvas = Image.new("RGB", (w + left + right, h + top + bottom), FILLS[mode])
        canvas.paste(rgb, (left, top))
        out = canvas
    else:
        raise ValueError(f"fill must be edge|black|white|gray, got {fill!r}")
    mask = build_mask(out.size, (left, top, w, h), int(feather))
    return out, mask


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    im = get_pimg(url, param("image"), tok, mode="RGB")
    out, mask = process(
        im,
        left=param("left", 0, cast=int),
        right=param("right", 0, cast=int),
        top=param("top", 0, cast=int),
        bottom=param("bottom", 0, cast=int),
        feather=param("feather", 32, cast=int),
        fill=param("fill", "edge"),
    )
    oid = post_pimg(url, out, tok)
    mid = post_pimg(url, mask, tok)
    print(f"[image-pad] image={param('image')} {im.size} -> {out.size} "
          f"(L{param('left', 0)} R{param('right', 0)} T{param('top', 0)} "
          f"B{param('bottom', 0)}) fill={param('fill', 'edge')} "
          f"-> image={oid} mask={mid}", flush=True)
    emit(image=oid, mask=mid, width=out.width, height=out.height)
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
