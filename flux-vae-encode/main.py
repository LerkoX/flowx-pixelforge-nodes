"""flux-vae-encode：FLUX VAE 编码，对标 ComfyUI VAE Encode。

自包含节点：服务端算子 flux.vae_encode 由 server_op.py 自注册。
图像 → 未 pack 归一化 LATENT，供 flux-sampler 以 denoise<1 做
img2img / hires-fix 两阶段。
"""
import time

from flowx_client import call_op, emit, ensure_plugin, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "flux.vae_encode", tok=tok)

    t0 = time.time()
    out = call_op(url, "flux.vae_encode",
                  {"vae": ref(param("vae")), "image": ref(param("image"))},
                  tok, timeout=param("job_timeout", 1800, int))
    print(f"[flux-vae-encode] image={param('image')} -> latent={out['latent']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(latent=out["latent"])


if __name__ == "__main__":
    main()
