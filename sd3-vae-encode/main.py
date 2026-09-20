"""sd3-vae-encode：SD3.5 VAE 编码（IMAGE → 16ch latent），供图生图（denoise<1）。

自包含节点：服务端算子 sd3.vae.encode 由 server_op.py 自注册（ensure_plugin）。
"""
import time

from flowx_client import call_op, emit, ensure_plugin, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "sd3.vae.encode", tok=tok)

    t0 = time.time()
    out = call_op(url, "sd3.vae.encode",
                  {"vae": ref(param("vae_ref")), "image": ref(param("image"))},
                  tok, timeout=600)
    print(f"[sd3-encode-img] image={param('image')} ({time.time()-t0:.1f}s) "
          f"-> latent={out['latent']}", flush=True)
    emit(latent=out["latent"])


if __name__ == "__main__":
    main()
