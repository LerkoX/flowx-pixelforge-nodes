"""video-depth：视频深度估计（Depth-Anything-V2-Base）。

为什么单独一个节点：
  - 之前深度只作为 video-person-mask 的内部步骤（depth_refine），
    用户看不到深度图、也没法复用到别处；
  - 本节点把深度**显式输出**（深度视频 + 首帧深度图），可单独预览/串联。

输出约定：近亮远暗（Depth-Anything 原生）。ControlNet depth 惯例是近暗，
需要时把 invert 打开。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "video.depth", tok=tok)

    video_id = param("video", "")
    if not video_id:
        raise RuntimeError("video-depth: video 必填（接 load-video）")

    # **bool 解析（坑）**：yaml 里 `invert: false` 传下来是字符串 "false"，
    # `bool("false")` == True（非空字符串恒真）——必须显式按字面量判定。
    _inv = str(param("invert", False)).strip().lower() in (
        "1", "true", "yes", "on")
    inputs = {"video": ref(video_id), "invert": _inv}
    t0 = time.time()
    jid = submit_job(url, {"name": "video.depth", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[video-depth] job={jid} invert={inputs['invert']}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[video-depth] -> video={out.get('video')} "
          f"depth={out.get('depth')} ({time.time()-t0:.1f}s)", flush=True)
    emit(video=out.get("video"), depth=out.get("depth"),
         fps=out.get("fps") or 16)


if __name__ == "__main__":
    main()
