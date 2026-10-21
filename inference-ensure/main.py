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
3. **复用感知**（1.3.0，`expect_resident`）：显存里已有同一模型时，下游 loader
   会命中服务端缓存（幂等，不新增显存占用），`min_vram_mb`「为新加载预留空间」的
   前提就不成立。列出本流水线要用、且期望复用的模型名后，只要它们**全部已常驻**，
   就跳过 `min_vram_mb` 检查（`require_headroom_mb` 与超订检查仍生效，采样
   working set 仍需余量）。注意：**同名模型不同 dtype/offload 旋钮 = 不同缓存键**
   （`名字|dtype=..|offload=..|t5=..`），旋钮不一致不会复用。

3. **复用感知**（1.3.0，`expect_resident`）：显存里已有同一模型时，下游 loader
   会命中服务端缓存（幂等，不新增显存占用），`min_vram_mb`「为新加载预留空间」的
   前提就不成立。列出本流水线要用、且期望复用的模型名后，只要它们**全部已常驻**，
   就跳过 `min_vram_mb` 检查（`require_headroom_mb` 与超订检查仍生效，采样
   working set 仍需余量）。注意：**同名模型不同 dtype/offload 旋钮 = 不同缓存键**
   （`名字|dtype=..|offload=..|t5=..`），旋钮不一致不会复用。

4. **FIFO 串行化闸门**（1.4.0，`queue_wait_seconds`）：多条流水线同时到达时，
   经 POST /leases/acquire 排队——只有队首且服务空闲（无其它活跃租约、无其它
   租约的 running/pending 任务）才放行。流水线结束后服务端按
   lease_idle_release_seconds（默认 300s 无活动）自动交还闸门，或由末尾的
   lease-release 节点立即交还。0 = 不排队（保持旧行为）。

5. **清场**（1.4.0，`evict_unused`，默认开）：轮到本执行后，把**本流水线不需要**
   的常驻模型全部淘汰（POST /model/unload keep=expect_resident）——上一轮流水线
   留下的模型即使没有任何租约占用也会一直占着显存，把新流水线顶进超订态
   （exec 839 事故：flux 三件套闲置 9.5h，reserved 12340MB > 物理 8192MB，
   新流水线 2.5s 硬失败）。pinned/其它租约持有的条目服务端自动跳过，不会误伤。
   清场后仍超订时（保留模型的分配器缓存没还驱动，exec 839 的实际形态）
   再补一把 POST /gc 收缩缓存（仅在持有闸门时做，/gc 会清整个对象仓库）；
   /gc 也收不动时（缓存与保留模型同段交织，exec 845 实测）升级为连保留模型
   一并卸载——保留让位于可运行性，下游 loader 会重载。

另把显存/常驻模型信息 emit 给下游（原实现只打日志），供画布与排障查看。
"""
import sys
import time

from flowx_client import emit, get_json, lease_id, param, post_json, token

# 模型文件名常见扩展名（比较常驻键与用户填写的名字时剥掉）
_EXTS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")


def _probe(url, tok):
    """返回 health_json。"""
    info = get_json(url, "/health", tok, timeout=10)
    return info


def _name_forms(raw):
    """一个名字 → 可比较的形态集合（小写、去组件前缀、去扩展名）。

    服务端常驻键形如 `poneDiffusion|dtype=fp16|offload=model|t5=0`、
    `cn:controlnet-x|dtype=fp16`；用户/流水线里通常只写文件名或裸模型名。
    """
    head = str(raw or "").strip().split("|", 1)[0]
    if not head:
        return set()
    forms = {head}
    for prefix in ("cn:", "vae:", "motion:"):
        if head.startswith(prefix):
            forms.add(head[len(prefix):])
    out = set()
    for name in forms:
        low = name.lower()
        out.add(low)
        for ext in _EXTS:
            if low.endswith(ext):
                out.add(low[: -len(ext)])
    return out


def _matches(token, key):
    """expect_resident 的一项是否命中某个常驻键。

    写**完整缓存键**（含 `|dtype=..`）时按精确匹配——旋钮不同就不算复用；
    写**裸名字/文件名**时只比较名字部分（含去 cn:/vae: 前缀与扩展名、忽略大小写），
    方便用户少抄旋钮，但需自行保证 dtype/offload 与下游 loader 一致。
    """
    t = str(token or "").strip()
    if not t:
        return False
    k = str(key).strip()
    if t.lower() == k.lower():
        return True
    if "|" in t:
        return False
    return bool(_name_forms(t) & _name_forms(k))


def main():
    url = param("service_url").rstrip("/")
    wait = param("wait_seconds", 60, int)
    min_vram = param("min_vram_mb", 0, int)
    min_headroom = param("require_headroom_mb", 0, int)
    raw_expect = param("expect_resident", "", str) or ""
    expect = [t.strip() for t in raw_expect.replace(";", ",").split(",") if t.strip()]
    tok = token()
    # 1.4.0：FIFO 排队 + 清场
    queue_wait = param("queue_wait_seconds", 3600, int)
    queue_poll = max(1, param("queue_poll_seconds", 5, int))
    evict_unused = str(param("evict_unused", "true", str) or "").strip().lower() \
        in ("1", "true", "yes", "on")
    if not expect:
        print("[ensure] WARNING: expect_resident 未填写（生图流水线必填）——"
              "清场将卸载**所有**未被保护的常驻模型（同流水线连跑会每次重载），"
              "且复用感知不生效。请按 loader 节点实际使用的模型填写"
              "（如 ponyDiffusionV6XL_v6StartWithThisOne 或 "
              "flux-unet:flux1-schnell-Q4_K_S.gguf,flux-clip:t5=...,flux-vae:vae）",
              file=sys.stderr, flush=True)

    t_start = time.time()
    queue_waited = 0.0
    gate_held = False
    # ---- FIFO 串行化闸门：轮到本执行再进场 ----
    # 无租约标识（命令行直跑/未注入 FLOWX_EXECUTION_ID）时跳过排队，保持旧行为
    if queue_wait > 0 and lease_id():
        qdeadline = t_start + queue_wait
        while True:
            try:
                res = post_json(url, "/leases/acquire", {}, tok, timeout=30)
            except RuntimeError as e:
                # 旧版服务端无 /leases/acquire（404/405/501）⇒ 跳过排队，
                # 保持 1.3.0 旧行为（不盲等 queue_wait_seconds 才失败）
                if any(f"HTTP {c}" in str(e) for c in (400, 404, 405, 501)):
                    print(f"[ensure] 服务端不支持排队闸门（{e}）——旧版服务？"
                          f"跳过排队（多流水线并发请升级服务端）", flush=True)
                    break
                if time.time() >= qdeadline:
                    raise
                print(f"[ensure] acquire 瞬态失败（重试）: {e}", flush=True)
                time.sleep(queue_poll)
                continue
            if res.get("acquired"):
                gate_held = True
                break
            queue_waited = time.time() - t_start
            if time.time() >= qdeadline:
                raise RuntimeError(
                    f"[ensure] 排队 {queue_waited:.0f}s 仍未轮到（queue_wait_seconds="
                    f"{queue_wait}）。当前：position={res.get('position')} "
                    f"ahead={res.get('ahead')} "
                    f"holders={[h.get('lease') for h in res.get('holders', [])]} "
                    f"active_jobs={len(res.get('active_jobs', []))}。"
                    f"办法：① 等前面的流水线跑完；② 调大 queue_wait_seconds；"
                    f"③ 确认无僵尸占用（监控页 / DELETE /leases/<id>）")
            pos = res.get("position")
            if pos:
                print(f"[ensure] 排队中：前面还有 {pos} 条流水线 "
                      f"{res.get('ahead')}（已等 {queue_waited:.0f}s）", flush=True)
            else:
                print(f"[ensure] 已到队首，等待服务空闲：holders="
                      f"{[h.get('lease') for h in res.get('holders', [])]} "
                      f"active_jobs={len(res.get('active_jobs', []))}"
                      f"（已等 {queue_waited:.0f}s）", flush=True)
            time.sleep(queue_poll)
        queue_waited = time.time() - t_start
        if queue_waited >= 1:
            print(f"[ensure] 获得闸门（排队 {queue_waited:.0f}s）", flush=True)

    # ---- 清场：淘汰本流水线不需要的常驻模型（evict_unused）----
    evicted, kept = "", ""
    if evict_unused:
        try:
            resident0 = (_probe(url, tok).get("resident_models") or [])
        except RuntimeError as e:
            resident0 = None
            print(f"[ensure] 清场前探测失败（跳过清场，交显存闸门判定）: {e}",
                  flush=True)
        if resident0:
            try:
                # 旧版服务端 /model/unload 无 keep 参数（会静默全卸，误伤想复用的
                # 模型）⇒ 先探能力：lease_idle_release_seconds 是随 keep 同版本
                # 引入的 policy 旋钮，存在即支持 keep
                pol = get_json(url, "/model/policy", tok, timeout=10)
                supports_keep = "lease_idle_release_seconds" in pol
            except RuntimeError:
                supports_keep = False
            try:
                if supports_keep:
                    res = post_json(url, "/model/unload",
                                    {"target": "all", "keep": expect},
                                    tok, timeout=300)
                    evicted = res.get("unloaded", "(none)")
                    kept = res.get("kept", "(none)")
                    if res.get("skipped_reasons"):
                        print(f"[ensure] 清场跳过：{res['skipped_reasons']}", flush=True)
                else:
                    # 回退：节点侧按 _matches 算出受害者，逐个按缓存键卸载
                    victims = [k for k in resident0
                               if not any(_matches(t, k) for t in expect)]
                    done = []
                    for k in victims:
                        r = post_json(url, "/model/unload", {"target": k},
                                      tok, timeout=300)
                        if r.get("unloaded") not in (None, "", "(none)"):
                            done.append(k)
                    evicted = ",".join(done) if done else "(none)"
                    kept_list = [k for k in resident0
                                 if any(_matches(t, k) for t in expect)]
                    kept = ",".join(kept_list) if kept_list else "(none)"
                    print("[ensure] 旧版服务端（无 keep 支持），已按节点侧匹配逐个清场",
                          flush=True)
                if evicted and evicted != "(none)":
                    print(f"[ensure] 清场：淘汰不需要的常驻模型 {evicted}"
                          f"（保留 {kept}）", flush=True)
            except RuntimeError as e:
                print(f"[ensure] WARNING: 清场失败（不阻断，交显存闸门判定）: {e}",
                      file=sys.stderr, flush=True)

    # ---- 缓存收缩：清场后仍超订（保留模型的分配器缓存顶着）→ /gc 一把 ----
    # 场景（exec 839 正是这样）：上一条流水线用的就是这批模型，keep 全保留，
    # 但采样留下的分配器缓存（几 GB，可回收）没还给驱动 ⇒ reserved 仍 > 物理
    # ⇒ 超订硬失败。empty_cache 即可回收，不需要卸载任何模型。
    # 只在持有闸门时做：/gc 会清整个对象仓库，无闸门时可能 nuking 别的执行的
    # 对象；持有闸门 ⇒ 无其它活跃执行，对象仓库里只剩历史垃圾，安全。
    if gate_held:
        try:
            occ = _probe(url, tok)
            if occ.get("vram_oversubscribed"):
                print("[ensure] 清场后仍超订（分配器缓存未还驱动）"
                      "→ POST /gc 收缩缓存", flush=True)
                post_json(url, "/gc", {}, tok, timeout=120)
                occ = _probe(url, tok)
            if occ.get("vram_oversubscribed"):
                # 升级处置（exec 845 实测）：/gc 的 empty_cache 只能释放**完全空闲**
                # 的分配器段；缓存与保留模型的权重在同一段里交织时一个字节也放不掉
                # （当时保留 pony 6952MB + 缓存 1576MB，/gc 后 reserved 纹丝不动，
                # 全卸后 8528→194MB 才恢复）。此时保留语义让位于可运行性：
                # 连保留模型一并卸载——反正下游 loader 会重载（慢 2-3 分钟，
                # 但比硬失败/分钟级每步的退化强）。
                print("[ensure] /gc 后仍超订（缓存与保留模型同段交织，收不动）"
                      "→ 连保留模型一并卸载（下游 loader 将重载）", flush=True)
                res = post_json(url, "/model/unload", {"target": "all"},
                                tok, timeout=300)
                print(f"[ensure] 超订升级卸载：{res.get('unloaded')}", flush=True)
                kept = "(none)"
        except RuntimeError as e:
            print(f"[ensure] WARNING: 超订收縮/升级卸载失败（交显存闸门判定）: {e}",
                  file=sys.stderr, flush=True)

    hits, miss, reuse_ok = [], [], False
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
            resident = info.get("resident_models") or []
            hits = [t for t in expect if any(_matches(t, k) for k in resident)]
            miss = [t for t in expect if t not in hits]
            # 复用感知：期望模型已全部常驻 ⇒ 下游 loader 命中缓存、不新增显存占用，
            # min_vram_mb「为新加载预留空间」的前提不成立 → 跳过该项（其余闸门照旧）
            reuse_ok = bool(expect) and not miss
            if info.get("vram_oversubscribed"):
                hard = True
                reason = (f"服务已处于物理显存超订态（分配额度 "
                          f"{info.get('vram_process_reserved_mb')}MB > 物理 "
                          f"{info.get('vram_total_mb')}MB，比例为 "
                          f"{info.get('vram_oversub_ratio')}）：CUDA 正在借主机内存，"
                          f"采样会退化到分钟级/步（WSL2 不报 OOM，等也没用）")
            elif min_vram > 0 and not reuse_ok and free is None:
                reason = f"服务未报告 vram_free_mb（无 CUDA？），无法校验 min_vram_mb={min_vram}"
            elif min_vram > 0 and not reuse_ok and free < min_vram:
                reason = (f"可用显存 {free}MB < min_vram_mb={min_vram}MB"
                          f"（常驻: {resident}, "
                          f"offload={info.get('offload_mode')}"
                          + (f"；expect_resident 未命中: {miss}" if expect else "")
                          + "）")
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
            reuse_hint = ""
            if miss:
                reuse_hint = (f"⑤ 若 {miss} 本应已常驻，检查 dtype/offload/use_t5 旋钮"
                              f"是否与常驻条目一致——旋钮不同会生成独立缓存键、不复用；")
            raise RuntimeError(
                f"[ensure] 显存闸门未通过（等待 {wait}s 后）：{reason}。"
                f"办法：① 换别的模型前先卸载（model-unload 节点 / POST /model/unload）；"
                f"② 降低分辨率/步数；③ 加载时用 offload=model|sequential；"
                f"④ 重启推理服务容器。{reuse_hint}"
                f"不建议放着不管——采样会走 host memory 兜底，慢到分钟级/步")
        print(f"[ensure] attempt {attempt}: {reason}, retry in 3s ...", flush=True)
        time.sleep(3)

    gpu = info.get("gpu_name", "unknown-gpu")
    resident = info.get("resident_models") or []
    fields = {
        "service_url": url,
        "status": info.get("status", "ok"),
        "gpu_name": gpu,
        "offload_mode": info.get("offload_mode", ""),
        "queue_waited_seconds": round(queue_waited, 1),
        "evicted_models": evicted or "(none)",
        "kept_models": kept or "(none)",
    }
    if expect:
        fields["expect_resident"] = ",".join(expect)
        fields["expect_resident_hit"] = f"{len(hits)}/{len(expect)}"
        fields["min_vram_checked"] = "false" if reuse_ok else "true"
        if reuse_ok:
            print(f"[ensure] 期望模型已全部常驻 {len(hits)}/{len(expect)} → 跳过 "
                  f"min_vram_mb={min_vram} 检查（复用缓存不占新显存）: {hits}", flush=True)
        elif miss:
            print(f"[ensure] 期望常驻未命中: {miss}（仍校验 min_vram_mb={min_vram}）",
                  flush=True)
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
