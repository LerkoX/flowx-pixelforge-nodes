"""wan-text-encode mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

print("[wan-text-encode][mock] ok")
emit(cond="mock-wan-cond")
