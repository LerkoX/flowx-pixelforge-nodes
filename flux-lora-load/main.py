"""flux-lora-load：给 FLUX GGUF transformer 挂 LoRA（对标 ComfyUI LoraLoader）。

自包含节点：服务端算子 flux.lora_apply 由 server_op.py 自注册（ensure_plugin）。
吃 flux-unet-load 的 MODEL，输出挂好补丁的同一 MODEL 引用，串在
flux-sampler / flux-detail-refine 之前即可生效（补丁留在常驻 transformer 上）。

走异步任务通道 POST /jobs + 轮询（LoRA 冷读盘 + PEFT 注入分钟级，且与
其它加载在服务端 FIFO 排队——同步 /op 会被隧道空闲超时掐断）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()
    lora = param("lora_name")
    strength = param("strength", 1.0, float)
    model = param("model")

    ensure_plugin(url, "flux.lora_apply", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.lora_apply", "inputs": {
        "model": ref(model),
        "lora": lora,
        "strength": strength,
    }}, tok)

    # 上报 job 标记（base + job_id）：Studio 取消时定向转发 /interrupt
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[flux-lora-load] job={jid} submitted (lora={lora} "
          f"strength={strength})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[flux-lora-load] {out.get('info')} ({time.time()-t0:.1f}s)",
          flush=True)
    emit(model=out["model"], info=out.get("info", ""))


if __name__ == "__main__":
    main()
