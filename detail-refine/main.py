"""detail-refine：ADetailer 式局部重绘（脸/手检测 → 裁剪放大 → img2img → 羽化贴回）。
首个自包含节点：server_op.py 随节点包自注册到推理服务（ensure_plugin）。

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
    detector = param("detector", "face")

    ensure_plugin(url, "detail.refine", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {
        "name": "detail.refine",
        "inputs": {"model": ref(param("model_ref")),
                   "pos": ref(param("positive")),
                   "neg": ref(param("negative")),
                   "image": ref(param("image")),
                   "detector": detector,
                   "conf": param("conf", 0.3, cast=float),
                   "padding": param("padding", 0.4, cast=float),
                   "denoise": param("denoise", 0.4, cast=float),
                   "steps": param("steps", 20, cast=int),
                   "cfg": param("cfg", 7.0, cast=float),
                   "sampler_name": param("sampler_name", "euler"),
                   "scheduler": param("scheduler", "normal"),
                   "seed": param("seed", -1, cast=int),
                   "guide_size": param("guide_size", 512, cast=int),
                   "max_targets": param("max_targets", 4, cast=int),
                   "feather": param("feather", 16, cast=int)},
    }, tok)
    # job 标记：Studio 取消/超时据此**定向转发** /interrupt（信号一旦输给
    # 执行器杀进程，远程任务就变孤儿；实测概率约 1/3）
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[detail-refine] detector={detector} refined={out.get('count', 0)} "
          f"image={param('image')} -> {out['image']}", flush=True)
    emit(image=out["image"], count=str(out.get("count", 0)))


if __name__ == "__main__":
    main()
