"""video-bg-erase"""
import time

from flowx_client import (emit, ensure_plugin, host_base, param, ref,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    actual_op = ensure_plugin(url, "video.bg_erase", tok=tok)
    inputs = {}
    inputs["video"] = param("video", "")
    inputs["mask"] = param("mask", "")
    inputs["dilate"] = param("dilate", 20, int)
    inputs["radius"] = param("radius", 15, int)
    inputs["method"] = param("method", "lama")
    inputs["color_match"] = param("color_match", 1, int)
    inputs["video"] = ref(inputs["video"])
    inputs["mask"] = ref(inputs["mask"])
    jid = submit_job(url, {"name": actual_op, "inputs": inputs}, tok)
    result = wait_job(url, jid, tok, timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    emit(video=out["video"], fps=out.get("fps") or 16,
         num_frames=out.get("num_frames") or 0)


if __name__ == "__main__":
    main()
