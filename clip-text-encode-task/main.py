"""clip-text-encode-task：CLIP Text Encode + PowerPaint task prompt。

task≠none 时服务端自动在文本后缀对应 P_ctxt/P_shape/P_obj task token
（embedding 来自 MODELS_DIR/task_prompts/powerpaint.safetensors）。
走任务式（/jobs+轮询）而非同步 /op：GPU 串行队列里可能排在长任务之后
（clip-text-encode 同款理由），短轮询不被隧道掐断。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    clip = param("clip_ref")
    text = param("text")
    task = param("task", "none")
    side = param("side", "positive")
    clip_skip = param("clip_skip", 0, cast=int)
    pbase = host_base(url)   # Studio 侧可达（job 标记用）
    tok = token()
    ensure_plugin(url, "sd.clip.encode_task", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {
        "name": "sd.clip.encode_task",
        "inputs": {"clip": ref(clip), "text": text, "task": task,
                   "side": side, "clip_skip": clip_skip},
    }, tok)
    # job 标记：Studio 取消/超时据此定向转发 /interrupt
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    preview = text if len(text) <= 60 else text[:60] + "..."
    print(f"[encode-task] clip={clip} task={task} side={side} "
          f"text=\"{preview}\" -> cond={out['cond']}",
          flush=True)
    emit(conditioning=out["cond"])


if __name__ == "__main__":
    main()
