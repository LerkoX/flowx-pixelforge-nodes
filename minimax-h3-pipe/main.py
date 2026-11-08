"""minimax-h3-pipe：H3 组件组装为完整管道（拆分装配版的收口）。

TRANSFORMER + TEXT_ENCODER + VAE → MODEL。秒级（纯引用拼装 + scheduler），
但仍走 job 通道统一隧道行为。
"""
from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "minimax.h3.pipe", tok=tok)
    inputs = {
        "transformer": ref(param("transformer")),
        "text_encoder": ref(param("text_encoder")),
        "vae": ref(param("vae")),
        "shift_video": param("shift_video", 12.0, float),
        "shift_audio": param("shift_audio", 3.0, float),
    }
    print(f"[h3-pipe] shift={inputs['shift_video']}/{inputs['shift_audio']}",
          flush=True)

    jid = submit_job(url, {"name": "minimax.h3.pipe", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 300, int), poll=2)
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[h3-pipe] done -> model={out['model']}", flush=True)
    emit(model=out["model"])


if __name__ == "__main__":
    main()
