"""video-flashvsr：FlashVSR v1.1 视频超分（一步蒸馏流式 SR）。

官方推荐 4×（480×832→1920×3328）；max_frames 可切段冒烟。
服务端算子 video.flashvsr 由 server_op.py 自注册（ensure_plugin）。
"""
import time

from flowx_client import (emit, emit_preview, emit_result_preview,
                          ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "video.flashvsr", tok=tok)

    inputs = {
        "video": ref(param("video")),
        "scale": param("scale", 4.0, float),
        "sparse_ratio": param("sparse_ratio", 2.0, float),
        "local_range": param("local_range", 11, int),
        "seed": param("seed", 0, int),
        "color_fix": param("color_fix", 1, int),
        "max_frames": param("max_frames", 0, int),
    }
    print(f"[video-flashvsr] scale={inputs['scale']} "
          f"sparse={inputs['sparse_ratio']} local_range={inputs['local_range']} "
          f"max_frames={inputs['max_frames']}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "video.flashvsr", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[video-flashvsr] job={jid} submitted", flush=True)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    outs = result["outputs"]
    video_id = outs["video"].get("id")
    fps = outs["fps"].get("value")
    n = outs["num_frames"].get("value")
    w = outs["width"].get("value")
    h = outs["height"].get("value")
    print(f"[video-flashvsr] done in {time.time()-t0:.1f}s "
          f"-> video={video_id} frames={n} {w}x{h} fps={fps}", flush=True)
    if video_id:
        emit_result_preview(pbase, video_id, tok, kind="videos")
    emit(video=video_id, fps=fps, num_frames=n, width=w, height=h)


if __name__ == "__main__":
    main()
