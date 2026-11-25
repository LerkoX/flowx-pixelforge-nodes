"""vram-policy mock：不读预设，直接吐 turbo 固定值。"""
from flowx_client import emit

print("[vram-policy][mock] turbo: tr=group(blocks=4), te=group(blocks=2)")
emit(tr_offload="group", tr_group_blocks=4,
     te_offload="group", te_group_blocks=2,
     policy="turbo: tr=group(blocks=4), te=group(blocks=2)")
