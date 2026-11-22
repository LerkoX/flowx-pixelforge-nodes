"""video-flow-lock：脸部光流锁定（FaceRefine-Plus 抗摆动）。

DIS 稠密光流逐帧把重绘脸部锁定回源视频五官位置，消除"脸在头上跳舞"
的帧间摆动/闪烁。接在 video-face-stitch 之后：
base=源视频，refined=stitch 输出。音轨从 refined 透传。

服务端算子 video.flow_lock 由 server_op.py 自注册（ensure_plugin）。
"""
import time

from flowx_client import (emit, emit_preview, emit_result_preview,
                          ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "video.flow_lock", tok=tok)

    inputs = {
        "base": ref(param("base")),
        "refined": ref(param("refined")),
        "sigma_frac": param("sigma_frac", 0.0067, float),
        "threshold": param("threshold", 4.0, float),
    }
    print(f"[video-flow-lock] sigma_frac={inputs['sigma_frac']} "
          f"threshold={inputs['threshold']}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "video.flow_lock", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[video-flow-lock] job={jid} submitted", flush=True)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    outs = result["outputs"]
    video_id = outs["video"].get("id")
    fps = outs["fps"].get("value")
    n = outs["num_frames"].get("value")
    print(f"[video-flow-lock] done in {time.time()-t0:.1f}s "
          f"-> video={video_id} frames={n} fps={fps}", flush=True)
    if video_id:
        emit_result_preview(pbase, video_id, tok, kind="videos")
    emit(video=video_id, fps=fps, num_frames=n)


if __name__ == "__main__":
    main()
