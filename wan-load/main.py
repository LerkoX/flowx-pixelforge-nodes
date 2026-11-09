"""wan-load：通义万相（Wan）视频模型装配，对标 ComfyUI 的 Wan 模型加载节点。

自包含节点：服务端算子 wan.load 由 server_op.py 自注册（ensure_plugin）。

三种管道（task 参数）：
- t2v      WanPipeline（纯文本 → 视频）
- i2v      WanImageToVideoPipeline（首帧 → 视频；同时给末帧即首尾帧插值）
- animate  WanAnimatePipeline（角色动画 / 角色替换，配 wan-animate 节点）

Wan2.2 是 MoE 双去噪器：transformer（高噪）+ transformer2（低噪），
boundary_ratio 处切换——transformer2 非空即启用。

14B bf16 ≈ 28GB > 24GB 显存 ⇒ transformer/TE 逐层 offload（CPU 常驻，
forward 时逐层上卡）。冷装配数分钟，走异步 POST /jobs + 轮询。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)   # Studio 侧可达地址（取消兜底用）
    tok = token()

    ensure_plugin(url, "wan.load", tok=tok)

    inputs = {
        "task": param("task", "t2v"),
        "transformer": param("transformer",
                             "wan2.1_t2v_14B_bf16.safetensors"),
        "transformer2": param("transformer2", ""),
        "text_encoder": param("text_encoder", "umt5_xxl_bf16.safetensors"),
        "vae": param("vae", "wan_2.1_vae.safetensors"),
        "image_encoder": param("image_encoder", ""),
        "lora": param("lora", ""),
        "lora_scale": param("lora_scale", 1.0, float),
        "dtype": param("dtype", "bf16"),
        "offload": param("offload", "sequential"),
        "boundary_ratio": param("boundary_ratio", 0.875, float),
        "flow_shift": param("flow_shift", 5.0, float),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "wan.load", "inputs": inputs}, tok)

    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[wan-load] job={jid} submitted (task={inputs['task']}, "
          f"transformer={inputs['transformer']}, "
          f"moe={bool(inputs['transformer2'])}, "
          f"lora={inputs['lora'] or 'none'})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[wan-load] -> model={out['model']} ({time.time()-t0:.1f}s)",
          flush=True)
    emit(model=out["model"])


if __name__ == "__main__":
    main()
