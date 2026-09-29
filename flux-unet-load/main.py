"""flux-unet-load：FLUX 扩散模型（transformer）加载，对标 ComfyUI UNETLoader。

自包含节点：服务端算子 flux.unet_load 由 server_op.py 自注册（ensure_plugin）。
GGUF Q4_K_S transformer 单独装载（fp16 compute），输出 MODEL 组件引用，
供 flux-sampler / flux-detail-refine 消费。幂等：已常驻则秒回。
"""
import time

from flowx_client import call_op, emit, ensure_plugin, param, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    transformer = param("transformer", "flux1-schnell-Q4_K_S.gguf")

    ensure_plugin(url, "flux.unet_load", tok=tok)

    t0 = time.time()
    out = call_op(url, "flux.unet_load", {"transformer": transformer},
                  tok, timeout=1800)
    print(f"[flux-unet-load] {transformer} -> model={out['model']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(model=out["model"])


if __name__ == "__main__":
    main()
