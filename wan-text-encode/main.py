"""wan-text-encode：Wan 文本编码（UMT5-XXL → COND）。

对齐 SD 的 clip-text-encode。Wan 有 CFG（guidance_scale>1 时生效），
所以正负两路都编好放进 cond —— gen 时不必再编码。
"""
from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "wan.encode", tok=tok)

    model_ref = param("model_ref")
    prompt = param("prompt")
    neg = param("negative_prompt", "")
    msl = param("max_sequence_length", 0, int)
    inputs = {
        "model": ref(model_ref),
        "prompt": prompt,
        "negative_prompt": neg,
        "max_sequence_length": msl,
    }
    prev = prompt if len(prompt) <= 60 else prompt[:60] + "..."
    print(f"[wan-text-encode] prompt=\"{prev}\" msl={msl}", flush=True)

    jid = submit_job(url, {"name": "wan.encode", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int), poll=5)
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[wan-text-encode] done -> cond={out['cond']}", flush=True)
    emit(cond=out["cond"])


if __name__ == "__main__":
    main()
