"""minimax-h3-transformer-load mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

print("[minimax-h3-transformer-load][mock] ok")
emit(transformer="mock-transformer-obj")
