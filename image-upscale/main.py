"""image-upscale：图像放大（hires.fix 前置），图像对象引用 → 放大后的图像对象引用。

为什么走任务式（/jobs+轮询）而不是同步 /op：算子本身秒级，但 GPU 串行队列
（EXEC_LOCK+FIFO）里可能排在数十分钟~数小时的任务之后——同步 /op 等待期零字节，
隧道必掐断（exec 743 实测本节点同步超时失败）；任务式把等待变成短轮询，
且带 job 标记 ⇒ Studio 取消时可定向中断，不留孤儿任务。"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)   # Studio 侧可达（job 标记用）
    tok = token()
    ensure_plugin(url, "image.upscale", tok=tok)
    scale = param("scale", 2.0, cast=float)
    width = param("width", 0, cast=int)
    height = param("height", 0, cast=int)
    method = param("method", "lanczos")

    t0 = time.time()
    jid = submit_job(url, {
        "name": "image.upscale",
        "inputs": {"image": ref(param("image")),
                   "scale": scale, "width": width, "height": height,
                   "method": method},
    }, tok)
    # job 标记：Studio 取消/超时据此**定向转发** /interrupt（信号一旦输给
    # 执行器杀进程，远程任务就变孤儿；实测概率约 1/3）
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[upscale] image={param('image')} scale={scale} "
          f"target={width}x{height} method={method} -> image={out['image']}",
          flush=True)
    emit(image=out["image"])


if __name__ == "__main__":
    main()
