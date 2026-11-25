"""vram-policy：显存策略数据节点（纯本地，零 GPU/零网络）。

把 offload 策略作为普通数据输出，下游 loader 节点用
{{ VramPolicy.tr_offload }} / {{ VramPolicy.tr_group_blocks }} 等引用，
画布上一处调参全流水线生效（2026-11-26 显存策略可调化）。
"""
from flowx_client import emit, param

# 预设 → (tr_offload, tr_group_blocks, te_offload, te_group_blocks)
# turbo    ~12GB 显存，最快（tr group×4 + te group×2，3090/4090 24G 实测余量足）
# balanced ~7GB（tr group×2 + te sequential）
# saver    ~3GB，最省最慢（双 sequential，每步全模型往返 PCIe）
_PRESETS = {
    "turbo": ("group", 4, "group", 2),
    "balanced": ("group", 2, "sequential", 0),
    "saver": ("sequential", 0, "sequential", 0),
}


def main():
    preset = str(param("preset", "turbo")).strip().lower()
    if preset == "custom":
        tr_off = str(param("tr_offload", "group")).strip() or "group"
        tr_gb = param("tr_group_blocks", 4, int)
        te_off = str(param("te_offload", "group")).strip() or "group"
        te_gb = param("te_group_blocks", 2, int)
    else:
        if preset not in _PRESETS:
            print(f"[vram-policy] 未知预设 {preset!r} → 回退 turbo", flush=True)
            preset = "turbo"
        tr_off, tr_gb, te_off, te_gb = _PRESETS[preset]

    summary = (f"{preset}: tr={tr_off}" +
               (f"(blocks={tr_gb})" if tr_off == "group" else "") +
               f", te={te_off}" +
               (f"(blocks={te_gb})" if te_off == "group" else ""))
    print(f"[vram-policy] {summary}", flush=True)
    emit(tr_offload=tr_off, tr_group_blocks=int(tr_gb),
         te_offload=te_off, te_group_blocks=int(te_gb),
         policy=summary)


main()
