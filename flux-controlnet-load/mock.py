"""flux-controlnet-load mock：透传伪 CONTROLNET id。"""
from flowx_client import emit, param

p = param("cn_path", "cn-union-pro")
oid = f"mock-flux-cn({p})"
print(f"[flux-controlnet-load][mock] path={p} -> {oid}")
emit(controlnet=oid, info=f"mock cn={p}")
