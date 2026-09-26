"""flux-clip-encode mock：不调服务，回显文本产出假 COND 引用与统计。"""
from flowx_client import emit, param

text = param("text", "mock prompt")
print(f"[flux-encode][mock] len={len(text)} text={text[:40]!r}")
emit(cond="mock-flux-cond",
     info=f"pe=(1,256,4096) pooled=(1,768) | len={len(text)}")
