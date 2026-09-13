"""inference-op：通用算子调用节点。调推理服务 POST /op，新算子无需新增节点包。

inputs_json 中对象引用写法：{"$id": "uuid"}；
workflow 接线时用参数绑定填充，如：
  inputs_json: '{"clip": {"$id": "{{ LoadCheckpoint.clip_ref }}"}, "text": "a cat"}'
"""
import json

from flowx_client import call_op, param, token


def main():
    url = param("service_url").rstrip("/")
    op_name = param("op_name")
    inputs_json = param("inputs_json", "{}")
    emit_keys = [k.strip() for k in param("emit_keys", "").split(",") if k.strip()]
    tok = token()

    try:
        inputs = json.loads(inputs_json)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"inputs_json is not valid JSON: {e}")

    flat = call_op(url, op_name, inputs, tok, timeout=3600)
    print(f"[op] {op_name} -> {json.dumps(flat, ensure_ascii=False)}", flush=True)

    print("```flowx-yaml")
    print(f"outputs_json: {json.dumps(flat, ensure_ascii=False)}")
    for k in emit_keys:
        if k in flat:
            print(f'{k}: "{flat[k]}"')
    print("```")


if __name__ == "__main__":
    main()
