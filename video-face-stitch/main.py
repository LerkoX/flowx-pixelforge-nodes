"""video-face-stitch：脸部修复回贴（PixelAI H3 FaceFix 末段）。

把重绘后的脸部特写按 boxes 逐帧缩放贴回原视频，高斯羽化消边缘；
可选脸部大小渐隐（Face Size Fade）；音轨从源视频透传。

服务端算子 video.face_stitch 由 server_op.py 自注册（ensure_plugin）。
"""
import time

from flowx_client import (emit, emit_preview, emit_result_preview,
                          ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "video.face_stitch", tok=tok)

    boxes_raw = param("boxes")
    print(f"[video-face-stitch] boxes len={len(boxes_raw or '')} "
          f"head={str(boxes_raw)[:120]!r}", flush=True)
    inputs = {
        "video": ref(param("video")),
        "crop_video": ref(param("crop_video")),
        "boxes": boxes_raw,
        "feather": param("feather", 24, int),
        "full_fix_below": param("full_fix_below", 0, int),
        "no_fix_above": param("no_fix_above", 0, int),
        "color_match": param("color_match", 1, int),
        "tone_mode": param("tone_mode", "freq_sep"),
        "lowpass_frac": param("lowpass_frac", 0.1, float),
        "detail_gain": param("detail_gain", 1.0, float),
    }
    print(f"[video-face-stitch] feather={inputs['feather']} "
          f"fade=({inputs['full_fix_below']},{inputs['no_fix_above']})",
          flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "video.face_stitch", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[video-face-stitch] job={jid} submitted", flush=True)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    outs = result["outputs"]
    video_id = outs["video"].get("id")
    fps = outs["fps"].get("value")
    n = outs["num_frames"].get("value")
    print(f"[video-face-stitch] done in {time.time()-t0:.1f}s "
          f"-> video={video_id} frames={n} fps={fps}", flush=True)
    if video_id:
        emit_result_preview(pbase, video_id, tok, kind="videos")
    emit(video=video_id, fps=fps, num_frames=n)


if __name__ == "__main__":
    main()
