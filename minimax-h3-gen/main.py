"""minimax-h3-gen：MiniMax-H3 联合视频+音频生成（fl2va 首末帧 / t2va 文生）。

自包含节点：服务端算子 minimax.h3.gen 由 server_op.py 自注册（ensure_plugin）。
guidance-distilled：无负提示、无 cfg；24fps 固定，5~15 秒（num_frames 自动对齐
17n+5）；挂 turbo LoRA 时 4 步出片。输出 VIDEO 对象（含 32kHz 立体声音轨，
服务端落盘时自动 mux 进 mp4）。

分钟级任务：走异步任务通道 POST /jobs + 轮询（同 video-gen 模式）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "minimax.h3.gen", tok=tok)

    inputs = {
        "model": ref(param("model_ref")),
        "prompt": param("prompt"),
        "num_frames": param("num_frames", 125, int),
        "seed": param("seed", -1, int),
        "steps": param("steps", 4, int),
        "preview_every": 0,
    }
    image_id = param("image", "")
    if image_id:
        inputs["image"] = ref(image_id)
    last_id = param("last_image", "")
    if last_id:
        inputs["last_image"] = ref(last_id)

    mode = "fl2va" if (image_id or last_id) else "t2va"
    print(f"[minimax-h3-gen] mode={mode} frames={inputs['num_frames']} "
          f"steps={inputs['steps']} seed={inputs['seed']}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "minimax.h3.gen", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[minimax-h3-gen] job={jid} submitted", flush=True)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 7200, int),
                      poll=param("poll_interval", 5, int))
    outs = result["outputs"]
    seed = outs["seed"].get("value")
    video_id = outs["video"].get("id")
    print(f"[minimax-h3-gen] done in {time.time()-t0:.1f}s "
          f"-> video={video_id} seed={seed}", flush=True)
    emit(video=video_id, seed=seed)


if __name__ == "__main__":
    main()
