"""video-face-crop：脸部追踪裁切（PixelAI H3 FaceFix 第一段）。

遮罩逐帧出脸部包围盒 → EMA 平滑防抖 → 方形裁切放大到 out_size。
输出脸部特写视频（接 H3Gen.inpaint_video）+ 全白遮罩（接 inpaint_mask）
+ boxes JSON（接 video-face-stitch）+ 脸部参考图。

服务端算子 video.face_crop 由 server_op.py 自注册（ensure_plugin）。
"""
import time

from flowx_client import (emit, emit_preview, emit_result_preview,
                          ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "video.face_crop", tok=tok)

    inputs = {
        "video": ref(param("video")),
        "mask": ref(param("mask")),
        "crop_factor": param("crop_factor", 2.5, float),
        "out_size": param("out_size", 768, int),
        "ema": param("ema", 0.75, float),
    }
    print(f"[video-face-crop] crop_factor={inputs['crop_factor']} "
          f"out_size={inputs['out_size']} ema={inputs['ema']}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "video.face_crop", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[video-face-crop] job={jid} submitted", flush=True)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    outs = result["outputs"]
    crop_id = outs["crop"].get("id")
    mask_id = outs["mask_white"].get("id")
    boxes = outs["boxes"].get("value")
    boxes_file = (outs.get("boxes_file") or {}).get("value") or boxes
    face_ref_id = outs["face_ref"].get("id")
    n = outs["num_frames"].get("value")
    fps = outs["fps"].get("value")
    fh = outs["face_h_max"].get("value")
    print(f"[video-face-crop] done in {time.time()-t0:.1f}s "
          f"-> crop={crop_id} frames={n} fps={fps} face_h_max={fh}",
          flush=True)
    if crop_id:
        emit_result_preview(pbase, crop_id, tok, kind="videos")
    emit(crop=crop_id, mask_white=mask_id, boxes=boxes,
         boxes_file=boxes_file, face_ref=face_ref_id,
         num_frames=n, fps=fps, face_h_max=fh)


if __name__ == "__main__":
    main()
