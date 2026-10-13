"""flux-vae-encode：FLUX VAE 编码，对标 ComfyUI VAE Encode。

自包含节点：服务端算子 flux.vae_encode 由 server_op.py 自注册。
图像 → 未 pack 归一化 LATENT，供 flux-sampler 以 denoise<1 做
img2img / hires-fix 两阶段。

为什么走任务式（/jobs+轮询）而不是同步 /op：算子本身 1~2s，但 GPU 串行队列
（engine.EXEC_LOCK + FIFO）里可能排在冷加载/采样之后干等数分钟——同步 /op 期间
连接零字节，隧道（cpolar）会掐断这类流式连接（exec 686 同因）；任务式把等待变成
短轮询（隧道抖动可容忍），且节点被取消时定向 interrupt，不留孤儿 GPU 任务。
"""
import time

from flowx_client import (emit, ensure_plugin, param, ref, submit_job,
                          token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    ensure_plugin(url, "flux.vae_encode", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {"name": "flux.vae_encode", "inputs": {
        "vae": ref(param("vae")),
        "image": ref(param("image")),
    }}, tok)
    print(f"[flux-vae-encode] job={jid} submitted "
          f"(image={param('image')})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[flux-vae-encode] image={param('image')} -> "
          f"latent={out['latent']} ({time.time()-t0:.1f}s)", flush=True)
    emit(latent=out["latent"])


if __name__ == "__main__":
    main()
