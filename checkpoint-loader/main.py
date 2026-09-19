"""checkpoint-loader：加载 checkpoint 到推理服务显存（幂等），输出 model/clip/vae 对象引用。

分钟级任务（冷加载需读盘数分钟）：走异步任务通道 POST /jobs + 轮询，
避免同步 /op 被隧道/代理的空闲超时掐断（同 motion-loader 模式）。"""
import time

from flowx_client import emit, get_json, param, submit_job, token, wait_job


def main():
    url = param("service_url").rstrip("/")
    ckpt = param("ckpt_name")
    tok = token()

    t0 = time.time()
    jid = submit_job(url, {"name": "checkpoint.load",
                           "inputs": {"ckpt": ckpt}}, tok)
    print(f"[loader] job={jid} submitted (ckpt={ckpt})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    resident = get_json(url, "/models", tok, timeout=30).get("resident_models", [])
    print(f"[loader] ckpt={ckpt} ({time.time()-t0:.1f}s) resident={resident}", flush=True)

    emit(model_ref=out["model"], clip_ref=out["clip"], vae_ref=out["vae"])


if __name__ == "__main__":
    main()
