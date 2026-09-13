"""inference-op mock：不调服务，回显伪输出。"""
import json

from flowx_client import param

op_name = param("op_name", "unknown")
inputs_json = param("inputs_json", "{}")
emit_keys = [k.strip() for k in param("emit_keys", "").split(",") if k.strip()]

fake = {k or "out": f"mock-{op_name}-{k or 'out'}" for k in (emit_keys or ["out"])}
print(f"[op][mock] {op_name} inputs={inputs_json}")

print("```flowx-yaml")
print(f"outputs_json: {json.dumps(fake, ensure_ascii=False)}")
for k, v in fake.items():
    print(f'{k}: "{v}"')
print("```")
