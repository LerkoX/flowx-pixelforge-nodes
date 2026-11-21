"""qwen-image-pipe：Qwen-Image 管道组装（分离式装配最后一步）。

自包含节点：服务端算子 qwen_image.pipe 由 server_op.py 自注册（ensure_plugin）。

把 qwen-image-transformer-load / qwen-image-te-load / qwen-image-vae-load 的
组件装成 QwenImagePipeline。组件常驻模型池 → 换 LoRA / 采样参数不必重装
39GB transformer。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)



def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "qwen_image.pipe", tok=tok)

    tr = param("transformer", "")
    te = param("text_encoder", "")
    va = param("vae", "")
    if not (tr and te and va):
        raise RuntimeError(
            "qwen-image-pipe: transformer / text_encoder / vae 三个组件必填")

    inputs = {
        "transformer": ref(tr),
        "text_encoder": ref(te),
        "vae": ref(va),
        "model": param("model", "Qwen-Image-2512"),
        "task": param("task", "t2i"),
        "lora": param("lora", ""),
        "lora_scale": param("lora_scale", 1.0, float),
        "max_sequence_length": param("max_sequence_length", 512, int),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "qwen_image.pipe", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[qwen-image-pipe] job={jid} task={inputs['task']} lora={inputs['lora'] or 'none'}",
          flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[qwen-image-pipe] -> model={out['model']} ({time.time()-t0:.1f}s)",
          flush=True)
    emit(model=out['model'])


if __name__ == "__main__":
    main()
