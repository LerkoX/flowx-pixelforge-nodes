"""vae-decode：VAE Decode，latent → 图像对象引用。"""
from flowx_client import call_op, emit, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    out = call_op(url, "vae.decode",
                  {"vae": ref(param("vae_ref")), "latent": ref(param("latent"))},
                  tok, timeout=600)
    print(f"[decode] -> image={out['image']}", flush=True)
    emit(image=out["image"])


if __name__ == "__main__":
    main()
