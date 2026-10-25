"""service-monitor：推理服务监控快照节点。

抓一次 `/health` + `/gpu` + `/leases` + `/jobs`，输出关键监控指标：
扁平标量（供下游节点做闸门判断，如显存不足则走分支）+ 一段压缩 JSON
快照（`snapshot`，回放态节点 UI 据此渲染）。

平时监控面板的实时刷新由 ui/node-widget.js 走设计期代理
（POST /api/v1/service-proxy）自行轮询，不依赖本脚本运行。
"""
import json
import time

from flowx_client import emit, get_json, param, token


def _gpu0(g):
    if isinstance(g, dict) and g.get("available") and g.get("gpus"):
        return g["gpus"][0] or {}
    return {}


def _num(v, default=-1):
    """标量输出兜底：None → default（输出契约要求整数，不能吐 None）。"""
    try:
        return int(round(float(v)))
    except (TypeError, ValueError):
        return default


def _short(s, n=18):
    s = str(s or "")
    return s if len(s) <= n else s[: n - 5] + "…" + s[-4:]


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    h = get_json(url, "/health", tok)
    g = get_json(url, "/gpu", tok)
    lz = get_json(url, "/leases", tok)
    jb = get_json(url, "/jobs", tok)

    gpu = _gpu0(g)
    total = _num(h.get("vram_total_mb"), 0)
    driver_free = h.get("vram_driver_free_mb")
    reserved = h.get("vram_process_reserved_mb")
    if driver_free is not None:
        used = max(0, total - _num(driver_free, 0))
    elif reserved is not None:
        used = _num(reserved, 0)
    else:
        used = 0

    leases = lz.get("leases") or []
    jobs = jb.get("jobs") or []
    running = [j for j in jobs if j.get("status") == "running"]
    pending = [j for j in jobs if j.get("status") == "pending"]
    residents = h.get("resident_details") or []
    resident_max = _num(h.get("max_resident_models"))

    # 租约 → 持有的常驻 key 反查表（常驻列表标注归属用）
    holder_of = {}
    for l in leases:
        for k in l.get("keys") or []:
            holder_of[k] = l.get("lease")

    active_jobs = sorted(running + pending,
                         key=lambda j: 0 if j.get("status") == "running" else 1)
    snapshot = {
        "ts": time.time(),
        "gpu_name": h.get("gpu_name") or "",
        "util": gpu.get("utilization_percent"),
        "temp": gpu.get("temperature_c"),
        "power": gpu.get("power_w"),
        "power_limit": gpu.get("power_limit_w"),
        "fan": gpu.get("fan_percent"),
        "throttle": [r for r in (gpu.get("throttle_reasons") or []) if r != "gpu_idle"],
        "vram": {
            "total": total,
            "alloc": h.get("vram_process_allocated_mb"),
            "reserved": reserved,
            "cache": h.get("vram_reclaimable_mb"),
            "driver_free": driver_free,
            "headroom": h.get("vram_headroom_mb"),
        },
        "oversubscribed": bool(h.get("vram_oversubscribed")),
        "saturated": bool(h.get("vram_saturated")),
        "resident_max": h.get("max_resident_models"),
        "residents": [
            {"key": d.get("key"), "arch": d.get("arch"),
             "vram_mb": d.get("est_vram_mb"), "idle": d.get("idle_seconds"),
             "pinned": bool(d.get("pinned")), "holder": holder_of.get(d.get("key"))}
            for d in residents[:20]
        ],
        "leases": [
            {"lease": l.get("lease"), "keys": len(l.get("keys") or []),
             "idle": l.get("idle_seconds")}
            for l in leases[:20]
        ],
        "queue": jb.get("queue"),
        "jobs": [
            {"id": _short(j.get("id"), 12), "ops": j.get("ops") or [],
             "status": j.get("status"), "queue_pos": j.get("queue_pos"),
             "percent": (j.get("progress") or {}).get("percent")}
            for j in active_jobs[:8]
        ],
    }
    snap_txt = json.dumps(snapshot, ensure_ascii=False, separators=(",", ":"))

    util = gpu.get("utilization_percent")
    temp = gpu.get("temperature_c")
    detail = "%s ｜ util %s ｜ %s ｜ vram %.1f/%.1fGB ｜ 常驻 %d/%s ｜ 租约 %d ｜ 队列 run%d/pen%d%s" % (
        h.get("gpu_name") or "GPU?",
        f"{util}%" if util is not None else "—",
        f"{temp}℃" if temp is not None else "—",
        used / 1024, total / 1024,
        len(residents), resident_max if resident_max >= 0 else "?",
        len(leases), len(running), len(pending),
        " ｜ ⚠️超订" if h.get("vram_oversubscribed") else "")

    print(f"[service-monitor] {detail}", flush=True)
    emit(ok="true", gpu_name=h.get("gpu_name") or "",
         gpu_util_percent=_num(util), gpu_temp_c=_num(temp),
         gpu_power_w=_num(gpu.get("power_w")),
         vram_used_mb=used, vram_total_mb=total,
         vram_headroom_mb=_num(h.get("vram_headroom_mb")),
         resident_count=len(residents), max_resident_models=resident_max,
         lease_count=len(leases),
         jobs_running=len(running), jobs_pending=len(pending),
         oversubscribed="true" if h.get("vram_oversubscribed") else "false",
         detail=str(detail)[:500], snapshot=snap_txt[:6000])


if __name__ == "__main__":
    main()
