"""wan-lora-load：Wan LoRA / 打光 LoRA 统一入口（配置校验 + 归一化）。

对应 ComfyUI 的 LoraLoader 位置，但**不做真合并**——合并发生在
wan-animate2 装配期（权重还在 CPU 内存态时，与块级 offload 无冲突）。
本节点只负责：校验文件存在、解析 `路径:强度` 语法、把配置串传给下游。

为什么单独拆节点：
  - 打光 LoRA（relight）是人物替换的**必开项**（让新角色光照匹配背景，
    治"像另一个图层"），但以前只能写在 wan-pipe 里、而独立装配又完全不读它
    → 实际一直是关的。拆成节点后可视化开关，且不会被装配路径吞掉。
"""
from flowx_client import (emit, ensure_plugin, host_base, param, ref,
                          submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    ensure_plugin(url, "wan.lora_load", tok=tok)

    lora = param("lora", "")
    lora_scale = param("lora_scale", 1.0, float)
    relight = param("relight_lora", "")
    relight_scale = param("relight_scale", 1.0, float)
    model_id = param("model", "")

    inputs = {"lora": lora, "lora_scale": lora_scale,
              "relight_lora": relight, "relight_scale": relight_scale}
    if model_id:
        inputs["model"] = ref(model_id)

    print(f"[wan-lora-load] lora={lora or '(无)'} relight={relight or '(无)'}",
          flush=True)
    jid = submit_job(url, {"name": "wan.lora_load", "inputs": inputs}, tok)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 600, int), poll=2)
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    cfg = out.get("lora_config", "")
    print(f"[wan-lora-load] -> {cfg[:160]}", flush=True)
    emit(lora_config=cfg)


if __name__ == "__main__":
    main()
