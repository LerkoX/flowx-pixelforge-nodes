"""image-composite：前景贴到底图（对标 ComfyUI Image Composite Masked），纯客户端 PIL。

mask 可选：给定时按蒙版贴（蒙版尺寸不符前景时自动缩放到前景尺寸）；
缺省时整张前景贴到 (x,y)。越界部分自动裁掉（PIL paste 语义）。
"""
from flowx_client import emit, param, token
from image_io import get_pimg, post_pimg, thumb_data_url


def process(background, foreground, mask=None, x=0, y=0):
    bg = background.convert("RGB").copy()
    fg = foreground.convert("RGB")
    if mask is not None:
        m = mask.convert("L")
        if m.size != fg.size:
            m = m.resize(fg.size)
        bg.paste(fg, (x, y), m)
    else:
        bg.paste(fg, (x, y))
    return bg


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    x = param("x", 0, cast=int)
    y = param("y", 0, cast=int)
    mask_id = param("mask", "")

    bg = get_pimg(url, param("background"), tok, mode="RGB")
    fg = get_pimg(url, param("foreground"), tok, mode="RGB")
    mask = get_pimg(url, mask_id, tok, mode="L") if mask_id else None

    out = process(bg, fg, mask, x, y)
    oid = post_pimg(url, out, tok)
    print(f"[composite] bg={param('background')} fg={param('foreground')} "
          f"mask={mask_id or '-'} at ({x},{y}) -> image={oid}", flush=True)
    emit(image=oid, width=out.width, height=out.height,
         thumb_b64=thumb_data_url(out))


if __name__ == "__main__":
    main()
