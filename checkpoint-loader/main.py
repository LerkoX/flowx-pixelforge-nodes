"""checkpoint-loader：加载 checkpoint 到推理服务显存（幂等），输出 model/clip/vae 对象引用。

分钟级任务（冷加载需读盘数分钟）：走异步任务通道 POST /jobs + 轮询，
避免同步 /op 被隧道/代理的空闲超时掐断（同 motion-loader 模式）。"""
import time

from flowx_client import emit, get_json, param, submit_job, token, wait_job


def main():
    url = param("service_url").rstrip("/")
    ckpt = param("ckpt_name")
    tok = token()

    # 性能旋钮（dtype/offload/use_t5）：auto/留空 = 继承服务端进程级环境变量；
    # 显式指定时作为模型级覆盖透传给 checkpoint.load——同模型不同旋钮组合
    # 是独立常驻条目（同一 LRU），SD1.5(none) 与 SD3.5(sequential) 可共存
    inputs = {"ckpt": ckpt}
    for knob in ("dtype", "offload", "use_t5"):
        v = param(knob, "auto").strip().lower()
        if v and v != "auto":
            inputs[knob] = v
    knobs = {k: v for k, v in inputs.items() if k != "ckpt"}

    t0 = time.time()
    jid = submit_job(url, {"name": "checkpoint.load",
                           "inputs": inputs}, tok)
    print(f"[loader] job={jid} submitted (ckpt={ckpt}, knobs={knobs or 'auto'})", flush=True)
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
