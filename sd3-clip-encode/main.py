"""sd3-clip-encode：SD3.5 CLIP 文本编码（双 CLIP-L/G + pooled），输出 COND 引用。

自包含节点：服务端算子 sd3.clip.encode 由 server_op.py 自注册（ensure_plugin）。
"""
from flowx_client import call_op, emit, ensure_plugin, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    clip = param("clip_ref")
    text = param("text")
    tok = token()

    ensure_plugin(url, "sd3.clip.encode", tok=tok)

    out = call_op(url, "sd3.clip.encode",
                  {"clip": ref(clip), "text": text}, tok, timeout=600)
    preview = text if len(text) <= 60 else text[:60] + "..."
    print(f"[sd3-encode] clip={clip} text=\"{preview}\" -> cond={out['cond']}",
          flush=True)
    emit(cond=out["cond"], info=out.get("info", ""))


if __name__ == "__main__":
    main()
