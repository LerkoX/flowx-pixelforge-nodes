"""instantid-apply mock：不调服务，透传伪 model/control id。"""
from flowx_client import emit, param

m = param("model", "model")
print(f"[instantid-apply][mock] model={m} weight={param('weight', '0.8')} "
      f"cn_strength={param('cn_strength', '0.8')}")
emit(model=f"mock-model-iid-{m}", control=f"mock-control-iid-{m}")
