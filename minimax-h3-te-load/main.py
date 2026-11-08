"""minimax-h3-te-load：H3 文本塔组件加载（拆分装配版）。"""
from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "minimax.h3.te_load", tok=tok)
    inputs = {
        "text_encoder": param(
            "text_encoder", "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
        "dtype": param("dtype", "bf16"),
        "offload": param("offload", "sequential"),
    }
    print(f"[h3-te-load] text_encoder={inputs['text_encoder']} "
          f"dtype={inputs['dtype']} offload={inputs['offload']}", flush=True)

    jid = submit_job(url, {"name": "minimax.h3.te_load", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int), poll=5)
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[h3-te-load] done -> text_encoder={out['text_encoder']}",
          flush=True)
    emit(text_encoder=out["text_encoder"])


if __name__ == "__main__":
    main()
