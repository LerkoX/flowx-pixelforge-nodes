"""flux-redux-apply mock：不调服务，透传伪 cond id。"""
from flowx_client import emit, param

c = param("cond", "mock")
oid = f"mock-cond-redux-{c}"
print(f"[flux-redux-apply][mock] cond={c} image={param('image', 'mock')} "
      f"strength={param('strength', '1.0')} -> {oid}")
emit(cond=oid, info="mock")
