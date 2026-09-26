"""flux-loader：FLUX.1（GGUF 量化）模型装配，输出 model 引用。

自包含节点：服务端算子 flux.load 由 server_op.py 自注册（ensure_plugin）。
GGUF Q4 量化 transformer + T5 + CLIP-L + VAE 原地装配 FluxPipeline
（低内存路线：GGUF from_single_file 裸读，无 safetensors→fp32→cast 峰值）。
幂等：同 transformer/t5/dtype 已装配则秒回缓存。
"""
import time

from flowx_client import (emit, ensure_plugin, param, submit_job, token,
                          wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "flux.load", tok=tok)

    transformer = param("transformer", "flux1-schnell-Q4_K_S.gguf")
    t5 = param("t5", "t5xxl-Q4_K_S.gguf")
    offload = param("offload", "auto")
    print(f"[flux-loader] transformer={transformer} t5={t5} "
          f"offload={offload}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.load", "inputs": {
        "transformer": transformer, "t5": t5, "offload": offload}}, tok)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    model_id = result["outputs"]["model"]["id"]
    print(f"[flux-loader] done in {time.time()-t0:.1f}s -> "
          f"model_ref={model_id}", flush=True)
    emit(model_ref=model_id)


if __name__ == "__main__":
    main()
