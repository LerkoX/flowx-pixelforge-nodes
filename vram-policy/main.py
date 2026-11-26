"""vram-policy：显存策略纯数据节点（单个 preset 下拉 → 输出 loader 配置）。"""
from flowx_client import emit, param

_PRESETS = {
    # 3090 实测：sequential ~7s/it 显存 ~3GB；group 被 diffusers 静默降级
    # （blocks>1 禁 stream）反而 26s/it —— saver 是当前最快档。
    "saver":    {"tr_offload": "sequential", "tr_group_blocks": 0,
                 "te_offload": "sequential", "te_group_blocks": 0},
    "balanced": {"tr_offload": "group",      "tr_group_blocks": 2,
                 "te_offload": "sequential", "te_group_blocks": 0},
    "turbo":    {"tr_offload": "group",      "tr_group_blocks": 4,
                 "te_offload": "group",      "te_group_blocks": 2},
}


def main():
    preset = str(param("preset", "saver")).lower()
    cfg = _PRESETS.get(preset)
    if cfg is None:
        raise ValueError(f"vram-policy: 未知 preset {preset!r}，"
                         f"可选：{sorted(_PRESETS)}")
    emit(**cfg, policy=f"{preset}(tr={cfg['tr_offload']}/"
                       f"{cfg['tr_group_blocks']},te={cfg['te_offload']}/"
                       f"{cfg['te_group_blocks']})")


if __name__ == "__main__":
    main()
