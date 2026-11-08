"""minimax-h3-te-load mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

print("[minimax-h3-te-load][mock] ok")
emit(text_encoder="mock-te-obj")
