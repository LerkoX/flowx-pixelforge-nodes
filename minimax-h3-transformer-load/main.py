"""minimax-h3-transformer-load：H3 transformer 组件加载（拆分装配版）。

输出 TRANSFORMER 对象（dict：model + is_ref 标记）→ minimax-h3-pipe 组装。
分钟级任务（int8 反量化 + LoRA 挂载），走异步 job 通道。
"""
from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "minimax.h3.transformer_load", tok=tok)
    inputs = {
        "transformer": param("transformer",
                             "minimax_h3_fl2va_int8_convrot.safetensors"),
        "lora": param("lora",
                      "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
        "lora_scale": param("lora_scale", 1.0, float),
        "dtype": param("dtype", "bf16"),
        "offload": param("offload", "sequential"),
    }
    print(f"[h3-tr-load] transformer={inputs['transformer']} "
          f"lora={inputs['lora'] or 'none'} dtype={inputs['dtype']} "
          f"offload={inputs['offload']}", flush=True)

    jid = submit_job(url, {"name": "minimax.h3.transformer_load",
                           "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int), poll=5)
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[h3-tr-load] done -> transformer={out['transformer']}", flush=True)
    emit(transformer=out["transformer"])


if __name__ == "__main__":
    main()
