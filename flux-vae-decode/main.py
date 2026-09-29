"""flux-vae-decode：FLUX VAE 解码，对标 ComfyUI VAE Decode。

自包含节点：服务端算子 flux.vae_decode 由 server_op.py 自注册。
消费 flux-sampler 的 LATENT（packed）或 flux-vae-encode 的 LATENT
（未 pack），unpack + 反归一化 + VAE 解码输出 IMAGE，接 save-image。
"""
import time

from flowx_client import call_op, emit, ensure_plugin, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "flux.vae_decode", tok=tok)

    t0 = time.time()
    out = call_op(url, "flux.vae_decode",
                  {"vae": ref(param("vae")), "latent": ref(param("latent"))},
                  tok, timeout=param("job_timeout", 1800, int))
    print(f"[flux-vae-decode] latent={param('latent')} -> image={out['image']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(image=out["image"])


if __name__ == "__main__":
    main()
