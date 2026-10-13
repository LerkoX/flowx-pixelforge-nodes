"""clip-text-encode：CLIP Text Encode，文本 → conditioning 对象引用。

按底模架构分派（服务端 clip.encode）：SD1.x 单编码器；SDXL 双 text encoder
+ pooled（采样时自动注入 added_cond_kwargs）。width/height/clip_skip 仅 SDXL
有意义（原始尺寸 → time_ids；clip_skip=-2 为 SDXL 常用），0 = 默认/auto。

为什么走任务式（/jobs+轮询）而不是同步 /op：算子本身秒级，但 GPU 串行队列
（EXEC_LOCK+FIFO）里可能排在数十分钟~数小时的任务之后——同步 /op 等待期零字节，
隧道必掐断（exec 743 实测本节点同步超时失败）；任务式把等待变成短轮询，
且带 job 标记 ⇒ Studio 取消时可定向中断，不留孤儿任务。"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    clip = param("clip_ref")
    text = param("text")
    width = param("width", 0, cast=int)
    height = param("height", 0, cast=int)
    clip_skip = param("clip_skip", 0, cast=int)
    pbase = host_base(url)   # Studio 侧可达（job 标记用）
    tok = token()
    ensure_plugin(url, "sd.clip.encode", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {
        "name": "sd.clip.encode",
        "inputs": {"clip": ref(clip), "text": text, "width": width,
                   "height": height, "clip_skip": clip_skip},
    }, tok)
    # job 标记：Studio 取消/超时据此**定向转发** /interrupt（信号一旦输给
    # 执行器杀进程，远程任务就变孤儿；实测概率约 1/3）
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    preview = text if len(text) <= 60 else text[:60] + "..."
    extra = (f" size={width}x{height} clip_skip={clip_skip}"
             if (width or height or clip_skip) else "")
    print(f"[encode] clip={clip} text=\"{preview}\"{extra} "
          f"-> cond={out['cond']}",
          flush=True)
    emit(conditioning=out["cond"])


if __name__ == "__main__":
    main()
