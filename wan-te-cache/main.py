"""wan-te-cache：UMT5-XXL 文本向量缓存。

首次运行：加载文本编码器 → 编码 prompt / negative → 存 safetensors；
后续同 prompt：直接返回缓存路径，**完全不加载 11GB 文本模型**。

为什么需要它：
  Wan-Animate 每次生成都要 UMT5 编码 prompt，而 UMT5-XXL 占 11GB
  （bf16）。在 24GB 卡上这 11GB 正是"跑不了 896×512 / 长视频"的主因。
  embedding 只与 prompt 有关、与画面无关，所以完全可以离线算一次反复用。

下游：把 embeds_path / negative_embeds_path 接到 wan-animate2 的
prompt_embeds_path / negative_embeds_path，算子会跳过文本模型加载。
"""
from flowx_client import (emit, ensure_plugin, host_base, param, ref,
                          submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    ensure_plugin(url, "wan.te_cache", tok=tok)

    prompt = param("prompt")
    neg = param("negative_prompt", "")
    msl = param("max_sequence_length", 0, int)
    te = param("text_encoder", "")

    inputs = {"prompt": prompt, "negative_prompt": neg,
              "max_sequence_length": msl, "text_encoder": te}
    prev = prompt if len(prompt) <= 50 else prompt[:50] + "..."
    print(f"[wan-te-cache] prompt=\"{prev}\" msl={msl}", flush=True)

    jid = submit_job(url, {"name": "wan.te_cache", "inputs": inputs}, tok)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int), poll=3)
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    hit = bool(out.get("cache_hit"))
    print(f"[wan-te-cache] {'命中缓存' if hit else '已编码并缓存'} "
          f"pos={out.get('embeds_path')}", flush=True)
    emit(embeds_path=out.get("embeds_path", ""),
         negative_embeds_path=out.get("negative_embeds_path", ""),
         cache_hit="1" if hit else "0")


if __name__ == "__main__":
    main()
