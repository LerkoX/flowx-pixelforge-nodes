"""inference-ensure：推理服务健康检查 + 显存闸门，等待就绪后输出 service_url。

两件事：
1. 就绪等待：服务没起来/重启中，按 wait_seconds 重试（原行为）；
2. 显存闸门（min_vram_mb>0，dev-plan §21.3 任务 4）：可用显存不足时**不放过**流水线，
   而是等到满足或超时明确失败。背景：8GB 卡上视频模型与图像模型同池常驻时，采样会
   静默退化到 17s/步、精修 127s/步（exec 361/362 实测），在首节点挡住比跑一小时才发现便宜。
另把显存/常驻模型信息 emit 给下游（原实现只打日志），供画布与排障查看。
"""
import sys
import time

from flowx_client import emit, get_json, param, token


def _probe(url, tok):
    """返回 (health_json, 闸门不满足的原因 or None)。"""
    info = get_json(url, "/health", tok, timeout=10)
    return info


def main():
    url = param("service_url").rstrip("/")
    wait = param("wait_seconds", 60, int)
    min_vram = param("min_vram_mb", 0, int)
    tok = token()

    deadline = time.time() + wait
    attempt = 0
    while True:
        attempt += 1
        info = None
        reason = None
        try:
            info = _probe(url, tok)
            free = info.get("vram_free_mb")
            if min_vram > 0:
                if free is None:
                    reason = f"服务未报告 vram_free_mb（无 CUDA？），无法校验 min_vram_mb={min_vram}"
                elif free < min_vram:
                    reason = (f"可用显存 {free}MB < min_vram_mb={min_vram}MB"
                              f"（常驻: {info.get('resident_models')}, "
                              f"offload={info.get('offload_mode')}）")
        except RuntimeError as e:
            reason = str(e)
        if reason is None:
            break
        if time.time() >= deadline:
            if info is None:
                print(f"[ensure] service not ready after {wait}s: {reason}", file=sys.stderr)
                raise RuntimeError(f"service not ready after {wait}s: {reason}")
            raise RuntimeError(
                f"[ensure] 显存闸门未通过（等待 {wait}s 后）：{reason}。"
                f"办法：① 换别的模型前先卸载（model-unload 节点 / POST /model/unload）；"
                f"② 降低分辨率/步数；③ 加载时用 offload=model|sequential；"
                f"④ 重启推理服务容器。不建议放着不管——采样会走 host memory 兜底，慢到分钟级/步")
        print(f"[ensure] attempt {attempt}: {reason}, retry in 3s ...", flush=True)
        time.sleep(3)

    gpu = info.get("gpu_name", "unknown-gpu")
    resident = info.get("resident_models") or []
    fields = {
        "service_url": url,
        "status": info.get("status", "ok"),
        "gpu_name": gpu,
        "offload_mode": info.get("offload_mode", ""),
    }
    if info.get("vram_free_mb") is not None:
        fields["vram_free_mb"] = info["vram_free_mb"]
        fields["vram_total_mb"] = info.get("vram_total_mb", "")
    fields["resident_models"] = ",".join(resident) if resident else "(empty)"
    print(f"[ensure] service ready: {url} gpu={gpu} "
          f"vram={info.get('vram_free_mb', '?')}/{info.get('vram_total_mb', '?')}MB "
          f"resident={resident} offload={info.get('offload_mode')} "
          f"degraded={info.get('degraded', {}).get('count', 0)}", flush=True)
    if info.get("vram_low"):
        print("[ensure] WARNING: vram_free 低于服务端 reserve，采样可能退化（见 /health vram_low）",
              file=sys.stderr, flush=True)
    emit(**fields)


if __name__ == "__main__":
    main()
