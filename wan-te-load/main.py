import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "wan.te_load", tok=tok)
    inputs = {
        "text_encoder": param("text_encoder", "umt5_xxl_bf16.safetensors"),
        "dtype": param("dtype", "bf16"),
        "offload": param("offload", "sequential"),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "wan.te_load", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[wan-te-load] job={jid} text_encoder={inputs['text_encoder']}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[wan-te-load] -> text_encoder={out['text_encoder']} ({time.time()-t0:.1f}s)", flush=True)
    emit(text_encoder=out['text_encoder'])


if __name__ == "__main__":
    main()
