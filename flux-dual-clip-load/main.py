"""flux-dual-clip-load：FLUX 双文本编码器加载，对标 ComfyUI DualCLIPLoader。

自包含节点：服务端算子 flux.dual_clip_load 由 server_op.py 自注册。
CLIP-L(bf16) + 双 tokenizer 立即装载；T5-XXL(GGUF) 编码相位才装载上卡
（内存错峰，12GB 内存盒关键设计）。输出 CLIP 组件束引用，供
flux-clip-encode 消费。幂等：已常驻则秒回。
"""
import time

from flowx_client import call_op, emit, ensure_plugin, param, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    t5 = param("t5", "t5xxl-Q4_K_S.gguf")

    ensure_plugin(url, "flux.dual_clip_load", tok=tok)

    t0 = time.time()
    out = call_op(url, "flux.dual_clip_load",
                  {"t5": t5, "dtype": param("dtype", "auto")},
                  tok, timeout=600)
    print(f"[flux-dual-clip-load] t5={t5} -> clip={out['clip']} "
          f"({time.time()-t0:.1f}s)", flush=True)
    emit(clip=out["clip"])


if __name__ == "__main__":
    main()
