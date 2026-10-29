"""mask-bbox-crop mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("image", "mock-image")
mid = param("mask", "mock-mask")
pad = param("padding", 64, cast=int)
tgt = param("target_long_side", 1536, cast=int)
print(f"[bbox-crop][mock] image={iid} mask={mid} padding={pad} target={tgt}")
emit(image=f"mock-bboxcrop-img-{iid}", mask=f"mock-bboxcrop-mask-{mid}",
     x=64, y=128, width=640, height=1280, crop_width=768, crop_height=1536)
