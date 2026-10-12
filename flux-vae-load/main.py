"""flux-vae-load：FLUX VAE 加载，对标 ComfyUI Load VAE。

自包含节点：服务端算子 flux.vae_load 由 server_op.py 自注册。
装载 FLUX 专用 AutoencoderKL（FLUX_DIR/vae，fp16），输出 VAE 组件引用，
供 flux-vae-decode / flux-vae-encode / flux-detail-refine 消费。幂等。

分钟级任务：VAE 本身装载很快，但它常与 flux-unet-load 同层并发提交，服务端
GPU 串行（engine.EXEC_LOCK）+ 任务 FIFO 会让本任务排在冷加载 UNet 之后——
走异步任务通道 POST /jobs + 轮询，等待期间不再需要长连接静默挂起，避免被
隧道/代理的空闲超时掐断（exec 686 事故；同 checkpoint-loader / motion-loader）。
"""
import time

from flowx_client import (emit, ensure_plugin, param, submit_job, token,
                          wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    name = param("name", "vae")

    ensure_plugin(url, "flux.vae_load", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.vae_load", "inputs": {
        "name": name,
        "dtype": param("dtype", "auto"),
    }}, tok)
    print(f"[flux-vae-load] job={jid} submitted (name={name})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[flux-vae-load] {name} -> vae={out['vae']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(vae=out["vae"])


if __name__ == "__main__":
    main()
