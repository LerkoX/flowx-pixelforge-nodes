"""flux-dual-clip-load：FLUX 双文本编码器加载，对标 ComfyUI DualCLIPLoader。

自包含节点：服务端算子 flux.dual_clip_load 由 server_op.py 自注册。
CLIP-L(bf16) + 双 tokenizer 立即装载；T5-XXL(GGUF) 编码相位才装载上卡
（内存错峰，12GB 内存盒关键设计）。输出 CLIP 组件束引用，供
flux-clip-encode 消费。幂等：已常驻则秒回。

分钟级任务：冷加载（CLIP-L 权重 + tokenizer 装配，实测 ~7 分钟）走异步任务
通道 POST /jobs + 轮询，避免同步 /op 被隧道/代理的空闲超时掐断（同
checkpoint-loader / motion-loader；exec 686 事故同族——长静默同步 /op 经
cpolar 隧道必被读超时打断）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)   # Studio 侧可达地址（取消兜底/预览帧中转用）
    tok = token()
    t5 = param("t5", "t5xxl-Q4_K_S.gguf")

    ensure_plugin(url, "flux.dual_clip_load", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.dual_clip_load", "inputs": {
        "t5": t5,
        "dtype": param("dtype", "auto"),
    }}, tok)

    # 上报 job 标记（base + job_id）：Studio 取消/超时时据此**定向转发**
    # POST /interrupt（dev-log §60：实测节点 SIGTERM 与执行器杀进程存在竞态，
    # 3 次取消只有 2 次由节点自己取消成功；带标记后由 Studio 兜底，确定性）。
    # 本节点无逐帧预览，url 仅占位（画布拉不到帧时静默无图，不影响执行）。
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[flux-dual-clip-load] job={jid} submitted (t5={t5})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[flux-dual-clip-load] t5={t5} -> clip={out['clip']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(clip=out["clip"])


if __name__ == "__main__":
    main()
