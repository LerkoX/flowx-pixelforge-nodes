"""cond-timestep-range：条件的时间步区间（分段提示词核心件）。

对标 ComfyUI ConditioningSetTimestepRange：让一条 conditioning 只在采样
进度 [start, end) 内生效，区间外整段跳过（避免不该生效的条件污染画面，
也省掉一次 UNet 前向 → 更快）。

典型用法：
- 分段提示词：前半程「起稿描述」、后半程「细节描述」，两条 cond 各设窗口后
  cond.combine 拼接
- 分阶段约束：只在后 30% 步注入风格条件
- 与 cond.set_area 叠加：区域 + 时间双重限定
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "cond.set_timestep_range", tok=tok)

    cond_id = param("cond", "")
    if not cond_id:
        raise RuntimeError("cond-timestep-range: cond 必填")

    inputs = {
        "cond": ref(cond_id),
        "start_percent": param("start_percent", 0.0, float),
        "end_percent": param("end_percent", 1.0, float),
    }

    t0 = time.time()
    jid = submit_job(url, {"name": "cond.set_timestep_range",
                           "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[cond-timestep-range] job={jid} window=["
          f"{inputs['start_percent']},{inputs['end_percent']})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 300, int),
                      poll=param("poll_interval", 2, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[cond-timestep-range] -> cond={out['cond']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(cond=out["cond"])


if __name__ == "__main__":
    main()
