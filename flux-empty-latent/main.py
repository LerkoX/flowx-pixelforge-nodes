"""flux-empty-latent：FLUX 空 Latent，对标 ComfyUI Empty Latent Image。

自包含节点：服务端算子 flux.empty_latent 由 server_op.py 自注册。
只携带尺寸/批次信息（samples=None），初始噪声由 flux-sampler 按 seed 生成。
宽高须 16 倍数。输出 LATENT 束供 flux-sampler 消费。
"""
from flowx_client import call_op, emit, ensure_plugin, param, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    width = param("width", 768, int)
    height = param("height", 768, int)
    batch = param("batch_size", 1, int)

    ensure_plugin(url, "flux.empty_latent", tok=tok)

    out = call_op(url, "flux.empty_latent",
                  {"width": width, "height": height, "batch_size": batch},
                  tok, timeout=60)
    print(f"[flux-empty-latent] {width}x{height} batch={batch} "
          f"-> latent={out['latent']}", flush=True)
    emit(latent=out["latent"])


if __name__ == "__main__":
    main()
