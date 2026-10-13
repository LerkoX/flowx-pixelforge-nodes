"""flux-vae-decode：FLUX VAE 解码，对标 ComfyUI VAE Decode。

自包含节点：服务端算子 flux.vae_decode 由 server_op.py 自注册。
消费 flux-sampler 的 LATENT（packed）或 flux-vae-encode 的 LATENT
（未 pack），unpack + 反归一化 + VAE 解码输出 IMAGE，接 save-image。

为什么走任务式（/jobs+轮询）而不是同步 /op：算子本身 1~2s，但 GPU 串行队列
（engine.EXEC_LOCK + FIFO）里可能排在冷加载/采样之后干等数分钟——同步 /op 期间
连接零字节，隧道（cpolar）会掐断这类流式连接（exec 686 同因）；任务式把等待变成
短轮询（隧道抖动可容忍），且节点被取消时定向 interrupt，不留孤儿 GPU 任务。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)   # Studio 侧可达地址（取消兜底/预览帧中转用）
    tok = token()

    ensure_plugin(url, "flux.vae_decode", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.vae_decode", "inputs": {
        "vae": ref(param("vae")),
        "latent": ref(param("latent")),
    }}, tok)

    # 上报 job 标记（base + job_id）：Studio 取消/超时时据此**定向转发**
    # POST /interrupt（dev-log §60：实测节点 SIGTERM 与执行器杀进程存在竞态，
    # 3 次取消只有 2 次由节点自己取消成功；带标记后由 Studio 兜底，确定性）。
    # 本节点无逐帧预览，url 仅占位（画布拉不到帧时静默无图，不影响执行）。
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[flux-vae-decode] job={jid} submitted "
          f"(latent={param('latent')})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[flux-vae-decode] latent={param('latent')} -> "
          f"image={out['image']} ({time.time()-t0:.1f}s)", flush=True)
    emit(image=out["image"])


if __name__ == "__main__":
    main()
