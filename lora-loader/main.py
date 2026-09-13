"""lora-loader：LoRA 加载，给 MODEL 挂增量补丁（可多个串联叠加），输出新 MODEL/CLIP 引用。"""
from flowx_client import call_op, emit, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    model = param("model_ref")
    lora = param("lora_name")
    strength = param("strength", 1.0, float)
    tok = token()

    out = call_op(url, "lora.apply",
                  {"model": ref(model), "lora": lora, "strength": strength},
                  tok, timeout=600)
    print(f"[lora] {lora}@{strength} applied -> model={out['model']}", flush=True)
    emit(model_ref=out["model"], clip_ref=out["clip"])


if __name__ == "__main__":
    main()
