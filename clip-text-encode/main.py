"""clip-text-encode：CLIP Text Encode，文本 → conditioning 对象引用。

按底模架构分派（服务端 clip.encode）：SD1.x 单编码器；SDXL 双 text encoder
+ pooled（采样时自动注入 added_cond_kwargs）。width/height/clip_skip 仅 SDXL
有意义（原始尺寸 → time_ids；clip_skip=-2 为 SDXL 常用），0 = 默认/auto。
"""
from flowx_client import call_op, emit, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    clip = param("clip_ref")
    text = param("text")
    width = param("width", 0, cast=int)
    height = param("height", 0, cast=int)
    clip_skip = param("clip_skip", 0, cast=int)
    tok = token()

    out = call_op(url, "clip.encode",
                  {"clip": ref(clip), "text": text, "width": width,
                   "height": height, "clip_skip": clip_skip}, tok, timeout=300)
    preview = text if len(text) <= 60 else text[:60] + "..."
    extra = f" size={width}x{height} clip_skip={clip_skip}" if (width or height
                                                                 or clip_skip) else ""
    print(f"[encode] clip={clip} text=\"{preview}\"{extra} -> cond={out['cond']}",
          flush=True)
    emit(conditioning=out["cond"])


if __name__ == "__main__":
    main()
