"""vae-encode：VAE Encode，图像对象引用 → latent 对象引用（图生图入口）。"""
from flowx_client import call_op, emit, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    out = call_op(url, "vae.encode",
                  {"vae": ref(param("vae_ref")), "image": ref(param("image"))},
                  tok, timeout=600)
    print(f"[encode] image={param('image')} -> latent={out['latent']}", flush=True)
    emit(latent=out["latent"])


if __name__ == "__main__":
    main()
