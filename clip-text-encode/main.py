"""clip-text-encode：CLIP Text Encode，文本 → conditioning 对象引用。"""
from flowx_client import call_op, emit, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    clip = param("clip_ref")
    text = param("text")
    tok = token()

    out = call_op(url, "clip.encode",
                  {"clip": ref(clip), "text": text}, tok, timeout=300)
    preview = text if len(text) <= 60 else text[:60] + "..."
    print(f"[encode] clip={clip} text=\"{preview}\" -> cond={out['cond']}", flush=True)
    emit(conditioning=out["cond"])


if __name__ == "__main__":
    main()
