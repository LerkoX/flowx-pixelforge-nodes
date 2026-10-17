"""inference-ensure：推理服务健康检查 + 显存闸门，等待就绪后输出 service_url。

两件事：
1. 就绪等待：服务没起来/重启中，按 wait_seconds 重试（原行为）；
2. 显存闸门（dev-plan §21.3 任务 4）：可用显存不足时**不放过**流水线，
   而是等到满足或超时明确失败。背景：8GB 卡上视频模型与图像模型同池常驻时，采样会
   静默退化到 17s/步、精修 127s/步（exec 361/362 实测），在首节点挡住比跑一小时才发现便宜。
   - `min_vram_mb`（乐观口径 vram_free_mb = 驱动余量 + 可回收缓存）
   - **`require_headroom_mb`（B1，exec 795）：物理口径余量** =
     vram_total_mb − vram_process_reserved_mb。两个口径的区别已在实测中踩过：
     WSL2/WDDM 下 CUDA 超订**不报 OOM**，而是借主机内存按页取用 ⇒ 采样
     170~210s/步（干净态 40s/步）而 OOM 自愈护栏全程未触发。
   - **超订即拦**（同上）：服务报 `vram_oversubscribed`（分配额度 > 物理显存，
     兜底的硬证据）时直接失败，不管 wait 多久 —— 它自己不会好，只能
     model-unload / 重启容器。
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
    min_headroom = param("require_headroom_mb", 0, int)
    tok = token()

    deadline = time.time() + wait
    attempt = 0
    while True:
        attempt += 1
        info = None
        reason = None
        hard = False   # True = 等下去也没用（超订态自己不会好），立即失败
        try:
            info = _probe(url, tok)
            free = info.get("vram_free_mb")
            room = info.get("vram_headroom_mb")
            if info.get("vram_oversubscribed"):
                hard = True
                reason = (f"服务已处于物理显存超订态（分配额度 "
                          f"{info.get('vram_process_reserved_mb')}MB > 物理 "
                          f"{info.get('vram_total_mb')}MB，比例为 "
                          f"{info.get('vram_oversub_ratio')}）：CUDA 正在借主机内存，"
                          f"采样会退化到分钟级/步（WSL2 不报 OOM，等也没用）")
            elif min_vram > 0 and free is None:
                reason = f"服务未报告 vram_free_mb（无 CUDA？），无法校验 min_vram_mb={min_vram}"
            elif min_vram > 0 and free < min_vram:
                reason = (f"可用显存 {free}MB < min_vram_mb={min_vram}MB"
                          f"（常驻: {info.get('resident_models')}, "
                          f"offload={info.get('offload_mode')}）")
            elif min_headroom > 0 and room is None:
                reason = (f"服务未报告 vram_headroom_mb（旧版服务？），"
                          f"无法校验 require_headroom_mb={min_headroom}")
            elif min_headroom > 0 and room < min_headroom:
                reason = (f"物理余量 {room}MB < require_headroom_mb={min_headroom}MB"
                          f"（reserved={info.get('vram_process_reserved_mb')}MB/"
                          f"物理 {info.get('vram_total_mb')}MB，常驻: "
                          f"{info.get('resident_models')}）")
        except RuntimeError as e:
            reason = str(e)
        if reason is None:
            break
        if hard and info is not None:
            raise RuntimeError(
                f"[ensure] 显存闸门未通过（硬失败，不重试）：{reason}。"
                f"办法：① POST /model/unload {{}} 或 model-unload 节点卸载常驻模型；"
                f"② 重启推理服务容器（清掉 host memory 兜底态，速度才会恢复）；"
                f"③ 确认没有第二条重流水线在跑（8GB 卡同时只跑一条）")
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
        # B1：物理口径（超订可见化）——画布/排障一眼看出是“真健康”还是“在借主机内存”
        for key in ("vram_headroom_mb", "vram_process_reserved_mb",
                    "vram_process_allocated_mb", "vram_budget_mb"):
            if info.get(key) is not None:
                fields[key] = info[key]
        fields["vram_oversubscribed"] = bool(info.get("vram_oversubscribed"))
    fields["resident_models"] = ",".join(resident) if resident else "(empty)"
    print(f"[ensure] service ready: {url} gpu={gpu} "
          f"vram={info.get('vram_free_mb', '?')}/{info.get('vram_total_mb', '?')}MB "
          f"物理余量={info.get('vram_headroom_mb', '?')}MB "
          f"oversub={info.get('vram_oversubscribed')} "
          f"resident={resident} offload={info.get('offload_mode')} "
          f"degraded={info.get('degraded', {}).get('count', 0)}", flush=True)
    if info.get("vram_oversubscribed"):
        print("[ensure] WARNING: 服务在物理超订态（CUDA 借主机内存），采样会退化到"
              "分钟级/步；建议 POST /model/unload 或重启容器（见 /health "
              "vram_oversubscribed）", file=sys.stderr, flush=True)
    elif info.get("vram_low"):
        print("[ensure] WARNING: vram_free 低于服务端 reserve，采样可能退化（见 /health vram_low）",
              file=sys.stderr, flush=True)
    emit(**fields)


if __name__ == "__main__":
    main()
