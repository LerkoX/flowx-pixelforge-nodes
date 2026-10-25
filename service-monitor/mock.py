"""mock：不连服务的假监控数据（Studio「运行测试」/ UI 预览用）。"""
import json
import time


def run(params):
    snapshot = {
        "ts": time.time(),
        "gpu_name": "MockGPU 12G",
        "util": 42, "temp": 55, "power": 128, "power_limit": 170, "fan": 46,
        "throttle": [],
        "vram": {"total": 12288, "alloc": 5200, "reserved": 6100,
                 "cache": 900, "driver_free": 5800, "headroom": 4300},
        "oversubscribed": False, "saturated": False,
        "resident_max": 2,
        "residents": [
            {"key": "flux-vae:vae", "arch": "flux", "vram_mb": 335,
             "idle": 12.0, "pinned": False, "holder": None},
            {"key": "ponyDiffusionV6XL_v6StartWithThisOne", "arch": "sdxl",
             "vram_mb": 6800, "idle": 95.0, "pinned": False, "holder": "mock-lease-1"},
        ],
        "leases": [{"lease": "mock-lease-1", "keys": 1, "idle": 95.0}],
        "queue": {"running": True, "running_title": "sample", "running_seconds": 8.0,
                  "pending": 1, "head_wait_seconds": 3.0},
        "jobs": [
            {"id": "job-a1", "ops": ["sample"], "status": "running",
             "queue_pos": None, "percent": 35},
            {"id": "job-b2", "ops": ["vae.decode"], "status": "pending",
             "queue_pos": 1, "percent": None},
        ],
    }
    return {
        "ok": "true", "gpu_name": "MockGPU 12G",
        "gpu_util_percent": 42, "gpu_temp_c": 55, "gpu_power_w": 128,
        "vram_used_mb": 6488, "vram_total_mb": 12288, "vram_headroom_mb": 4300,
        "resident_count": 2, "max_resident_models": 2,
        "lease_count": 1, "jobs_running": 1, "jobs_pending": 1,
        "oversubscribed": "false",
        "detail": "MockGPU 12G ｜ util 42% ｜ 55℃ ｜ vram 6.3/12.0GB ｜ 常驻 2/2 ｜ 租约 1 ｜ 队列 run1/pen1",
        "snapshot": json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")),
    }
