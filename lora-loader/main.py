"""lora-loader：LoRA 加载，给 MODEL 挂增量补丁（可多个串联叠加），输出新 MODEL/CLIP 引用。

走异步任务通道 POST /jobs + 轮询（同 motion-loader 模式）：sequential offload /
隧道环境下挂载补丁可能触发组件搬运，同步 /op 会被代理空闲超时掐断。"""
from flowx_client import emit, param, ref, submit_job, token, wait_job


def main():
    url = param("service_url").rstrip("/")
    model = param("model_ref")
    lora = param("lora_name")
    strength = param("strength", 1.0, float)
    tok = token()

    jid = submit_job(url, {"name": "lora.apply",
                           "inputs": {"model": ref(model), "lora": lora,
                                      "strength": strength}}, tok)
    print(f"[lora] job={jid} submitted ({lora}@{strength})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[lora] {lora}@{strength} applied -> model={out['model']}", flush=True)
    emit(model_ref=out["model"], clip_ref=out["clip"])


if __name__ == "__main__":
    main()
