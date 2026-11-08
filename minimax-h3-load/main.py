"""minimax-h3-load：MiniMax-H3 联合视频+音频模型装配，对标 ComfyUI 模型加载节点。

自包含节点：服务端算子 minimax.h3.load 由 server_op.py 自注册（ensure_plugin）。
Comfy-Org int8 量化权重反量化 + 官方 config → diffusers MiniMaxH3ModularPipeline；
transformer / Qwen3VL-32B 文本塔逐层 offload（24GB 卡跑 bf16 的必由之路）。
可选 turbo LoRA（4 步出片）。幂等：已常驻则秒回。

分钟级任务：冷加载反量化 + 装配要数分钟，走异步任务通道 POST /jobs + 轮询
（同 flux-unet-load / video-gen 模式），避免同步 /op 被隧道空闲超时掐断。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)   # Studio 侧可达地址（取消兜底用）
    tok = token()

    ensure_plugin(url, "minimax.h3.load", tok=tok)

    inputs = {
        "transformer": param("transformer",
                             "minimax_h3_fl2va_int8_convrot.safetensors"),
        "text_encoder": param("text_encoder",
                              "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"),
        "video_vae": param("video_vae",
                           "vae-diffusers"),
        "audio_vae": param("audio_vae",
                           "minimax_h3_audio_vae_official.safetensors"),
        "lora": param("lora", "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors"),
        "lora_scale": param("lora_scale", 1.0, float),
        "dtype": param("dtype", "bf16"),
        "offload": param("offload", "sequential"),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "minimax.h3.load", "inputs": inputs}, tok)

    # 上报 job 标记（base + job_id）：Studio 取消时定向转发 /interrupt
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[minimax-h3-load] job={jid} submitted "
          f"(transformer={inputs['transformer']}, lora={inputs['lora'] or 'none'})",
          flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[minimax-h3-load] -> model={out['model']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(model=out["model"])


if __name__ == "__main__":
    main()
