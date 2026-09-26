"""flux-clip-encode：FLUX 文本编码（T5-XXL + CLIP-L），输出 COND 引用。

自包含节点：服务端算子 flux.encode 由 server_op.py 自注册（ensure_plugin）。
T5 与 transformer 分时复用内存（两相不叠加）：release_t5=1（默认）编码后
立即卸载 T5。首次编码建 T5 权重缓存约 6 分钟，此后缓存命中约 3 分钟。
"""
from flowx_client import (emit, ensure_plugin, param, ref, submit_job, token,
                          wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "flux.encode", tok=tok)

    text = param("text")
    jid = submit_job(url, {"name": "flux.encode", "inputs": {
        "model": ref(param("model_ref")),
        "text": text,
        "max_seq": param("max_seq", 256, int),
        "release_t5": param("release_t5", 1, int),
    }}, tok)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    outs = result["outputs"]
    cond_id = outs["cond"]["id"]
    info = outs.get("info", {}).get("value", "")
    preview = text if len(text) <= 60 else text[:60] + "..."
    print(f"[flux-encode] text=\"{preview}\" -> cond={cond_id} {info}",
          flush=True)
    emit(cond=cond_id, info=info)


if __name__ == "__main__":
    main()
