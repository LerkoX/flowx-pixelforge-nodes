"""flux-redux-apply：FLUX Redux 参考图注入（对标 ComfyUI Apply Style Model）。

自包含节点：服务端算子 flux.redux_apply 由 server_op.py 自注册。
消费 flux-clip-encode 的 COND + load-image 的参考图 IMAGE，输出注入参考图
语义的新 COND（729 个 Redux token 拼在文本序列前），接 flux-sampler 的
cond。strength 控制参考图影响强度（乘性缩放投影 embeds）。

分钟级任务：冷路径要装载 SigLIP + Redux 适配器（实测 138~141s 静默无字节），
走异步任务通道 POST /jobs + 轮询，避免同步 /op 被隧道/代理的空闲超时掐断
（同 checkpoint-loader / motion-loader / flux-*-load@1.2.0）。
"""
import time

from flowx_client import (emit, ensure_plugin, param, ref, submit_job, token,
                          wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "flux.redux_apply", tok=tok)

    strength = param("strength", 1.0, float)
    t0 = time.time()
    jid = submit_job(url, {"name": "flux.redux_apply", "inputs": {
        "cond": ref(param("cond")),
        "image": ref(param("image")),
        "strength": strength,
        "reducer": param("reducer", "flux1-redux-dev.safetensors"),
        "vision": param("vision", "siglip-so400m-patch14-384"),
        "release": param("release", 0, int),
    }}, tok)
    print(f"[flux-redux-apply] job={jid} submitted (strength={strength})",
          flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[flux-redux-apply] strength={strength} -> cond={out['cond']} "
          f"{out.get('info', '')} ({time.time()-t0:.1f}s)", flush=True)
    emit(cond=out["cond"], info=out.get("info", ""))


if __name__ == "__main__":
    main()
