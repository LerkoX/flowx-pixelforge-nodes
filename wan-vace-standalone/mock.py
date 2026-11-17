"""mock：不真跑 GPU。"""
from flowx_client import emit, param

print(f"[wan-vace][mock] steps={param('steps', 30)} skipped", flush=True)
emit(video="mock-wan-vace-video", seed=42, fps=16)
