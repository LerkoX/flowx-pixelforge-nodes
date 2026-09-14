"""ksampler：KSampler 采样，消费正/反 conditioning 与 latent，输出采样后 latent。

实时预览：运行/续跑时 Studio 会向节点注入 FLOWX_CALLBACK_URL_PUBLIC /
FLOWX_AUTH_TOKEN 环境变量，本节点将其透传给推理服务的 sample 算子，服务端在
采样循环中逐步 POST latent 预览帧（JPEG）回 Studio，画布节点上实时渲染
（等效 ComfyUI 采样器的实时预览）。未注入时静默跳过。
"""
import os
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
    # 实时预览：把 Studio 注入的回调地址透传给推理服务。
    # 预览帧由【远程推理服务】推送，必须用 PUBLIC 地址（FLOWX_CALLBACK_URL
    # 在 local 执行器下是回环地址，远程服务不可达）
    callback_url = (os.environ.get("FLOWX_CALLBACK_URL_PUBLIC")
                    or os.environ.get("FLOWX_CALLBACK_URL", ""))
    if callback_url:
        inputs["preview_callback_url"] = callback_url
        inputs["preview_token"] = os.environ.get("FLOWX_AUTH_TOKEN", "")
        inputs["preview_every"] = param("preview_every", 1, int)
        print(f"[ksampler] preview push enabled (every {inputs['preview_every']} step)", flush=True)
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
