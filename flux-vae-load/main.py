"""flux-vae-load：FLUX VAE 加载，对标 ComfyUI Load VAE。

自包含节点：服务端算子 flux.vae_load 由 server_op.py 自注册。
装载 FLUX 专用 AutoencoderKL（FLUX_DIR/vae，fp16），输出 VAE 组件引用，
供 flux-vae-decode / flux-vae-encode / flux-detail-refine 消费。幂等。
"""
import time

from flowx_client import call_op, emit, ensure_plugin, param, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    name = param("name", "vae")

    ensure_plugin(url, "flux.vae_load", tok=tok)

    t0 = time.time()
    out = call_op(url, "flux.vae_load",
                  {"name": name, "dtype": param("dtype", "auto")},
                  tok, timeout=600)
    print(f"[flux-vae-load] {name} -> vae={out['vae']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(vae=out["vae"])


if __name__ == "__main__":
    main()
