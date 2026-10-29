"""pose-gate mock：不调服务，透传伪骨架图 id（跳过人工确认等待）。"""
from flowx_client import emit, param

iid = f"mock-posegate-{param('image', 'img')}"
print(f"[pose-gate][mock] hand={param('include_hand', 'true')} "
      f"face={param('include_face', 'false')} -> {iid}")
emit(image=iid, gate_action="confirm")
