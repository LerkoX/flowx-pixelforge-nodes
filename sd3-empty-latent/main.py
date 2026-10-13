"""sd3-empty-latent：SD3.5 空 Latent（16 通道），输出 LATENT 引用。

自包含节点：服务端算子 sd3.latent.empty 由 server_op.py 自注册（ensure_plugin）。

为什么走任务式（/jobs+轮询）而不是同步 /op：算子本身秒级，但 GPU 串行队列
（EXEC_LOCK+FIFO）里可能排在数十分钟~数小时的任务之后——同步 /op 等待期零字节，
隧道必掐断（exec 743 实测本节点同步超时失败）；任务式把等待变成短轮询，
且带 job 标记 ⇒ Studio 取消时可定向中断，不留孤儿任务。"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)   # Studio 侧可达（job 标记用）
    tok = token()
    w = param("width", 512, int)
    h = param("height", 512, int)
    batch = param("batch_size", 1, int)

    ensure_plugin(url, "sd3.latent.empty", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {
        "name": "sd3.latent.empty",
        "inputs": {"width": w, "height": h, "batch_size": batch},
    }, tok)
    # job 标记：Studio 取消/超时据此**定向转发** /interrupt（信号一旦输给
    # 执行器杀进程，远程任务就变孤儿；实测概率约 1/3）
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[sd3-latent] {w}x{h} batch={batch} -> latent={out['latent']}",
          flush=True)
    emit(latent=out["latent"], info=out.get("info", ""))


if __name__ == "__main__":
    main()
