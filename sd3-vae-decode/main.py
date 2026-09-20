"""sd3-vae-decode：SD3.5 VAE 解码（16ch latent → IMAGE）。

自包含节点：服务端算子 sd3.vae.decode 由 server_op.py 自注册（ensure_plugin）。
"""
import time

from flowx_client import call_op, emit, ensure_plugin, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "sd3.vae.decode", tok=tok)

    t0 = time.time()
    out = call_op(url, "sd3.vae.decode",
                  {"vae": ref(param("vae_ref")), "latent": ref(param("latent"))},
                  tok, timeout=600)
    print(f"[sd3-decode] latent={param('latent')} ({time.time()-t0:.1f}s) "
          f"-> image={out['image']}", flush=True)
    emit(image=out["image"])


if __name__ == "__main__":
    main()
