"""flux-loader mock：不调服务，回显参数产出假 model 引用。"""
from flowx_client import emit, param

transformer = param("transformer", "flux1-schnell-Q4_K_S.gguf")
t5 = param("t5", "t5xxl-Q4_K_S.gguf")
print(f"[flux-loader][mock] transformer={transformer} t5={t5}")
emit(model_ref="mock-flux-model")
