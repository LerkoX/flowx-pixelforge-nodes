"""flux-controlnet-load：装载 FLUX ControlNet（Union-Pro 多模态，depth/canny 等）。

自包含节点：服务端算子 flux.controlnet_load 由 server_op.py 自注册
（ensure_plugin）。输出 CONTROLNET 引用，接 flux-mask-refine 的 controlnet
输入即可在 Fill 局部重绘时注入结构约束（锁体型轮廓）。

走异步任务通道 POST /jobs + 轮询（冷读盘 6.6GB 分钟级，同步 /op 会被
隧道空闲超时掐断）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()
    path = param("cn_path", "cn-union-pro", str)
    dtype = param("dtype", "fp16", str)
    qmode = param("quantize", "int8", str)

    ensure_plugin(url, "flux.controlnet_load", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.controlnet_load", "inputs": {
        "path": path,
        "dtype": dtype,
        "quantize": qmode,
    }}, tok)

    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[flux-controlnet-load] job={jid} submitted (path={path} "
          f"dtype={dtype} quantize={qmode})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[flux-controlnet-load] {out.get('info')} ({time.time()-t0:.1f}s)",
          flush=True)
    emit(controlnet=out["controlnet"], info=out.get("info", ""))


if __name__ == "__main__":
    main()
