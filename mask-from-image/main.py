"""mask-from-image：从图像提取蒙版（对标 ComfyUI Mask From Image）。

mask 约定 = 内容为灰度的 IMAGE 对象（上传后服务端转 RGB，灰度保留，
下游读回 convert("L") 即还原）。
"""
from flowx_client import emit, param, token
from image_io import get_pimg, post_pimg, thumb_data_url

_CHANNELS = {"red": "R", "green": "G", "blue": "B"}


def process(im, channel="luminance"):
    if channel == "luminance":
        return im.convert("RGB").convert("L")
    if channel not in _CHANNELS:
        raise RuntimeError(
            f"channel must be red|green|blue|luminance, got {channel!r}")
    return im.convert("RGB").getchannel(_CHANNELS[channel])


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    channel = param("channel", "luminance")

    im = get_pimg(url, param("image"), tok)
    out = process(im, channel)
    oid = post_pimg(url, out, tok)
    print(f"[mask-from-image] image={param('image')} channel={channel} "
          f"-> mask={oid} {out.width}x{out.height}", flush=True)
    emit(mask=oid, width=out.width, height=out.height,
         thumb_b64=thumb_data_url(out))


if __name__ == "__main__":
    main()
