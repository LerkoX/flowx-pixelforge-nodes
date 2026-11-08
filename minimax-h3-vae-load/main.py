"""minimax-h3-vae-load：H3 双 VAE 组件加载（拆分装配版）。"""
from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "minimax.h3.vae_load", tok=tok)
    inputs = {
        "video_vae": param("video_vae", "vae-diffusers"),
        "audio_vae": param("audio_vae",
                           "minimax_h3_audio_vae_official.safetensors"),
        "dtype": param("dtype", "bf16"),
        "device": param("device", "cuda"),
    }
    print(f"[h3-vae-load] video_vae={inputs['video_vae']} "
          f"audio_vae={inputs['audio_vae']} device={inputs['device']}",
          flush=True)

    jid = submit_job(url, {"name": "minimax.h3.vae_load", "inputs": inputs},
                     tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int), poll=5)
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[h3-vae-load] done -> vae={out['vae']}", flush=True)
    emit(vae=out["vae"])


if __name__ == "__main__":
    main()
