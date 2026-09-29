"""flux-clip-encode mock：生成伪 COND id。"""
from flowx_client import emit, param

text = param("text", "mock")
preview = text if len(text) <= 30 else text[:30] + "..."
oid = f"mock-flux-cond-{abs(hash(text)) % 100000}"
print(f"[flux-clip-encode][mock] text=\"{preview}\" -> cond={oid}")
emit(cond=oid, info="mock")
