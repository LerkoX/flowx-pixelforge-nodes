"""ksampler：KSampler 采样，消费正/反 conditioning 与 latent，输出采样后 latent。"""
import time

from flowx_client import call_op, emit, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    inputs = {
        "model": ref(param("model_ref")),
        "pos": ref(param("positive")),
        "neg": ref(param("negative")),
        "latent": ref(param("latent")),
        "seed": param("seed", -1, int),
        "steps": param("steps", 20, int),
        "cfg": param("cfg", 7.0, float),
        "sampler_name": param("sampler_name", "euler"),
        "denoise": param("denoise", 1.0, float),
    }
    print(f"[ksampler] steps={inputs['steps']} cfg={inputs['cfg']} "
          f"sampler={inputs['sampler_name']} seed={inputs['seed']} "
          f"denoise={inputs['denoise']}", flush=True)

    t0 = time.time()
    out = call_op(url, "sample", inputs, tok, timeout=3600)
    print(f"[ksampler] done in {time.time()-t0:.1f}s -> latent={out['latent']} "
          f"seed={out['seed']}", flush=True)
    emit(latent=out["latent"], seed=out["seed"])


if __name__ == "__main__":
    main()
