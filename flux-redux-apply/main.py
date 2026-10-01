"""flux-redux-apply：FLUX Redux 参考图注入（对标 ComfyUI Apply Style Model）。

自包含节点：服务端算子 flux.redux_apply 由 server_op.py 自注册。
消费 flux-clip-encode 的 COND + load-image 的参考图 IMAGE，输出注入参考图
语义的新 COND（729 个 Redux token 拼在文本序列前），接 flux-sampler 的
cond。strength 控制参考图影响强度（乘性缩放投影 embeds）。
"""
import time

from flowx_client import call_op, emit, ensure_plugin, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "flux.redux_apply", tok=tok)

    strength = param("strength", 1.0, float)
    t0 = time.time()
    out = call_op(url, "flux.redux_apply",
                  {"cond": ref(param("cond")),
                   "image": ref(param("image")),
                   "strength": strength,
                   "reducer": param("reducer", "flux1-redux-dev.safetensors"),
                   "vision": param("vision", "siglip-so400m-patch14-384"),
                   "release": param("release", 0, int)},
                  tok, timeout=param("job_timeout", 900, int))
    print(f"[flux-redux-apply] strength={strength} -> cond={out['cond']} "
          f"{out.get('info', '')} ({time.time()-t0:.1f}s)", flush=True)
    emit(cond=out["cond"], info=out.get("info", ""))


if __name__ == "__main__":
    main()
