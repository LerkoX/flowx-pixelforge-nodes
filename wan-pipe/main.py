"""wan-pipe：Wan 管道组装（分离式装配的最后一步）。

自包含节点：服务端算子 wan.pipe 由 server_op.py 自注册（ensure_plugin）。

把 wan-transformer-load / wan-te-load / wan-vae-load（+ wan-image-encoder-load）
的组件装成可生成管道。task 决定管道类型：
- t2v      WanPipeline
- i2v      WanImageToVideoPipeline（生成时给 last_image 即首尾帧 FLF2V）
- animate  WanAnimatePipeline（角色动画 / 替换，配 wan-animate 节点）

组件常驻模型池 → 换 VAE、flow_shift、LoRA 都不必重装 28GB transformer。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "wan.pipe", tok=tok)

    tr = param("transformer", "")
    te = param("text_encoder", "")
    va = param("vae", "")
    if not (tr and te and va):
        raise RuntimeError(
            "wan-pipe: transformer / text_encoder / vae 三个组件必填")

    def _bool(name, default="0"):
        return param(name, default) in ("1", "true", "True", "yes", "on")

    inputs = {
        "task": param("task", "t2v"),
        "transformer": ref(tr),
        "text_encoder": ref(te),
        "vae": ref(va),
        "boundary_ratio": param("boundary_ratio", 0.875, float),
        "flow_shift": param("flow_shift", 5.0, float),
        "scheduler": param("scheduler", "unipc"),
        "lora": param("lora", ""),
        "lora_scale": param("lora_scale", 1.0, float),
        "vae_tiling": _bool("vae_tiling"),
        "vae_slicing": _bool("vae_slicing"),
        "attention": param("attention", "default"),
        "max_sequence_length": param("max_sequence_length", 512, int),
    }
    ie = param("image_encoder", "")
    if ie:
        inputs["image_encoder"] = ref(ie)

    t0 = time.time()
    jid = submit_job(url, {"name": "wan.pipe", "inputs": inputs}, tok)

    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[wan-pipe] job={jid} task={inputs['task']} "
          f"lora={inputs['lora'] or 'none'} "
          f"flow_shift={inputs['flow_shift']} attn={inputs['attention']}",
          flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[wan-pipe] -> model={out['model']} ({time.time()-t0:.1f}s)",
          flush=True)
    emit(model=out["model"])


if __name__ == "__main__":
    main()
