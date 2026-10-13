"""flux-unet-load：FLUX 扩散模型（transformer）加载，对标 ComfyUI UNETLoader。

自包含节点：服务端算子 flux.unet_load 由 server_op.py 自注册（ensure_plugin）。
GGUF Q4_K_S transformer 单独装载（fp16 compute），输出 MODEL 组件引用，
供 flux-sampler / flux-detail-refine 消费。幂等：已常驻则秒回。

分钟级任务：冷加载要读盘数分钟，且同层其它加载会在服务端 FIFO 排队——
走异步任务通道 POST /jobs + 轮询，避免同步 /op 被隧道/代理的「空闲无字节」
超时掐断（同 checkpoint-loader / motion-loader 模式；exec 686 事故：同步
/op 经 cpolar 隧道读超时，UnetLoad/VaeLoad 双失败并因层屏障拖垮整条流水线）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)   # Studio 侧可达地址（取消兜底/预览帧中转用）
    tok = token()
    transformer = param("transformer", "flux1-schnell-Q4_K_S.gguf")

    ensure_plugin(url, "flux.unet_load", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.unet_load", "inputs": {
        "transformer": transformer,
        "dtype": param("dtype", "auto"),
    }}, tok)

    # 上报 job 标记（base + job_id）：Studio 取消/超时时据此**定向转发**
    # POST /interrupt（dev-log §60：实测节点 SIGTERM 与执行器杀进程存在竞态，
    # 3 次取消只有 2 次由节点自己取消成功；带标记后由 Studio 兜底，确定性）。
    # 本节点无逐帧预览，url 仅占位（画布拉不到帧时静默无图，不影响执行）。
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[flux-unet-load] job={jid} submitted (transformer={transformer})",
          flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[flux-unet-load] {transformer} -> model={out['model']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(model=out["model"])


if __name__ == "__main__":
    main()
