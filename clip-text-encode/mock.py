"""clip-text-encode mock：生成伪 cond_id。"""
import hashlib

from flowx_client import emit, param

text = param("text", "")
cid = "mock-cond-" + hashlib.md5(text.encode()).hexdigest()[:12]
print(f"[encode][mock] text=\"{text[:50]}\" -> {cid}")
emit(conditioning=cid)
