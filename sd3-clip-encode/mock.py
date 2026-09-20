"""sd3-clip-encode mock：不调服务，回显文本产出假 COND 引用与统计。"""
from flowx_client import emit, param

text = param("text", "mock prompt")
print(f"[sd3-encode][mock] len={len(text)} text={text[:40]!r}")
emit(cond="mock-sd3-cond",
     info=f"embeds=(1,154,4096) norm=0.00 | pooled=(1,2048) norm=0.00 | len={len(text)}")
