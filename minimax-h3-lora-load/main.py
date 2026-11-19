"""minimax-h3-lora-load：H3 LoRA 独立挂载节点（配置校验型）"""
import time

from flowx_client import (emit, ensure_plugin, host_base, param, ref,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()
    actual_op = ensure_plugin(url, "h3.lora_load", tok=tok)
    inputs = {
        "lora": param("lora", ""),
        "lora_scale": param("lora_scale", 1.0, float),
    }
    m = param("model", "")
    if m:
        inputs["model"] = ref(m)
    jid = submit_job(url, {"name": actual_op, "inputs": inputs}, tok)
    result = wait_job(url, jid, tok, timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m2.get("id", m2.get("value"))
           for k, m2 in result.get("outputs", {}).items()}
    print(f"[h3-lora-load] -> {out.get('lora_config','')}", flush=True)
    emit(lora_config=out.get("lora_config", ""))


if __name__ == "__main__":
    main()
