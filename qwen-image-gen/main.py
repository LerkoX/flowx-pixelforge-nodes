"""qwen-image-gen：Qwen-Image 文生图（T2I）。

自包含节点：服务端算子 qwen_image.gen 由 server_op.py 自注册（ensure_plugin）。

true_cfg_scale>1 且给了 negative_prompt 才走 CFG（官方默认 4.0）；
挂 Lightning LoRA 时用 steps=4/8 + true_cfg_scale=1.0。
分钟级任务，走异步 POST /jobs + 轮询。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)



def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "qwen_image.gen", tok=tok)

    model_id = param("model", "")
    if not model_id:
        raise RuntimeError(
            "qwen-image-gen: model 必填（接 qwen-image-pipe 的 model 输出）")

    inputs = {
        "model": ref(model_id),
        "prompt": param("prompt", ""),
        "negative_prompt": param("negative_prompt", ""),
        "steps": param("steps", 50, int),
        "true_cfg_scale": param("true_cfg_scale", 4.0, float),
        "width": param("width", 1328, int),
        "height": param("height", 1328, int),
        "seed": param("seed", -1, int),
        "max_sequence_length": param("max_sequence_length", 0, int),
        "num_images": param("num_images", 1, int),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "qwen_image.gen", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[qwen-image-gen] job={jid} {inputs['width']}x{inputs['height']} "
          f"steps={inputs['steps']} cfg={inputs['true_cfg_scale']} "
          f"seed={inputs['seed']}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[qwen-image-gen] -> image={out['image']} seed={out['seed']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(image=out["image"], seed=out["seed"])


if __name__ == "__main__":
    main()
