"""video-pose-mask"""
import time

from flowx_client import (emit, ensure_plugin, host_base, param, ref,
                          submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    actual_op = ensure_plugin(url, "video.pose_mask", tok=tok)
    inputs = {}
    inputs["video"] = param("video", "")
    inputs["grow"] = param("grow", 40, int)
    inputs["feather"] = param("feather", 21, int)
    inputs["threshold"] = param("threshold", 10, int)
    inputs["limit_expand"] = param("limit_expand", 4, int)
    inputs["mosaic"] = param("mosaic", 0, int)
    _lm = param("limit_mask", "")
    if _lm:
        inputs["limit_mask"] = ref(_lm)
    inputs["video"] = ref(inputs["video"])
    jid = submit_job(url, {"name": actual_op, "inputs": inputs}, tok)
    result = wait_job(url, jid, tok, timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    emit(mask=out["mask"], fps=out.get("fps") or 16,
         num_frames=out.get("num_frames") or 0)


if __name__ == "__main__":
    main()
