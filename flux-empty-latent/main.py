"""flux-empty-latent：FLUX 空 Latent，对标 ComfyUI Empty Latent Image。

自包含节点：服务端算子 flux.empty_latent 由 server_op.py 自注册。
只携带尺寸/批次信息（samples=None），初始噪声由 flux-sampler 按 seed 生成。
宽高须 16 倍数。输出 LATENT 束供 flux-sampler 消费。

为什么走任务式（/jobs+轮询）而不是同步 /op：本算子 0.2s，但它在 GPU 串行队列
（engine.EXEC_LOCK + FIFO）里可能排在冷加载/采样之后——历史实测排队 60s，采样
期间可达 30min+；同步 /op 等待期连接零字节，隧道（cpolar）会掐断（exec 686 同因）。
任务式把等待变成短轮询（隧道抖动可容忍），且可取消，不留孤儿任务。
job_timeout 默认 1800：真机 wf96 实测冷加载 UNet 期间本节点排队 602s（还在排）
——600s 的旧默认会被自己的超时掐掉（exec 724），故与 loader 族对齐。
"""
import time

from flowx_client import (emit, ensure_plugin, param, submit_job, token,
                          wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    width = param("width", 768, int)
    height = param("height", 768, int)
    batch = param("batch_size", 1, int)

    ensure_plugin(url, "flux.empty_latent", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.empty_latent", "inputs": {
        "width": width,
        "height": height,
        "batch_size": batch,
    }}, tok)
    print(f"[flux-empty-latent] job={jid} submitted "
          f"({width}x{height} batch={batch})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 2, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[flux-empty-latent] {width}x{height} batch={batch} -> "
          f"latent={out['latent']} ({time.time()-t0:.1f}s)", flush=True)
    emit(latent=out["latent"])


if __name__ == "__main__":
    main()
