import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "wan.vae_load", tok=tok)
    inputs = {
        "vae": param("vae", "wan_2.1_vae.safetensors"),
        "device": param("device", "cuda"),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "wan.vae_load", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[wan-vae-load] job={jid} vae={inputs['vae']}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[wan-vae-load] -> vae={out['vae']} ({time.time()-t0:.1f}s)", flush=True)
    emit(vae=out['vae'])


if __name__ == "__main__":
    main()
