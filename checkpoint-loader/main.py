"""checkpoint-loader：加载 checkpoint 到推理服务显存（幂等），输出 model/clip/vae 对象引用。"""
import time

from flowx_client import call_op, emit, get_json, param, token


def main():
    url = param("service_url").rstrip("/")
    ckpt = param("ckpt_name")
    tok = token()

    t0 = time.time()
    out = call_op(url, "checkpoint.load", {"ckpt": ckpt}, tok, timeout=3600)
    resident = get_json(url, "/models", tok, timeout=30).get("resident_models", [])
    print(f"[loader] ckpt={ckpt} ({time.time()-t0:.1f}s) resident={resident}", flush=True)

    emit(model_ref=out["model"], clip_ref=out["clip"], vae_ref=out["vae"])


if __name__ == "__main__":
    main()
