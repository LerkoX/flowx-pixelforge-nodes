"""video-person-mask：SAM3 文本提示检测 + 全片追踪，输出人物遮罩视频。

多人精确编辑（H3 人物替换 inpaint_mask）的选区件：用自然语言点名
被替换者（如 "the woman in white dress"），SAM3 在首帧检测并逐帧
追踪，输出灰度遮罩视频（255=选中区）。

服务端算子 video.person_mask 由 server_op.py 自注册（ensure_plugin）。
模型 facebook/sam3（0.86B）常驻服务端模型管理器。
"""
import time

from flowx_client import (emit, emit_preview, emit_result_preview,
                          ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "video.person_mask", tok=tok)

    inputs = {
        "video": ref(param("video")),
        "prompt": param("prompt", "person"),
        "obj_pick": param("obj_pick", "all"),
        "dilate": param("dilate", 12, int),
        "max_frames": param("max_frames", 0, int),
    }
    print(f"[video-person-mask] prompt={inputs['prompt']!r} "
          f"obj_pick={inputs['obj_pick']} dilate={inputs['dilate']}",
          flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "video.person_mask", "inputs": inputs},
                     tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[video-person-mask] job={jid} submitted", flush=True)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    outs = result["outputs"]
    mask_id = outs["mask"].get("id")
    n = outs["num_frames"].get("value")
    w = outs["width"].get("value")
    h = outs["height"].get("value")
    objects = outs["objects"].get("value")
    print(f"[video-person-mask] done in {time.time()-t0:.1f}s "
          f"-> mask={mask_id} frames={n} {w}x{h} objects={objects}",
          flush=True)
    if mask_id:
        emit_result_preview(pbase, mask_id, tok, kind="videos")
    emit(mask=mask_id, num_frames=n, width=w, height=h, objects=objects)


if __name__ == "__main__":
    main()
