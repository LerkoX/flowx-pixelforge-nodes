"""flux-kontext-edit：FLUX.1-Kontext-dev 图像编辑节点"""
import time

from flowx_client import (emit, ensure_plugin, host_base, param, ref,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()
    actual_op = ensure_plugin(url, "flux.kontext_edit", tok=tok)
    inputs = {
        "model": ref(param("model")),
        "vae": ref(param("vae")),
        "cond": ref(param("cond")),
        "image": ref(param("image")),
        "seed": param("seed", -1, int),
        "steps": param("steps", 20, int),
        "guidance": param("guidance", 3.5, float),
        "denoise": param("denoise", 0.7, float),
    }
    jid = submit_job(url, {"name": actual_op, "inputs": inputs}, tok)
    result = wait_job(url, jid, tok, timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    emit(image=out["image"])


if __name__ == "__main__":
    main()
