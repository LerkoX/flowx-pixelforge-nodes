"""catvton-load：装载 CatVTON 换装管道（vton.catvton_load，服务端插件）。

把 SD1.5-inpainting 基座 + CatVTON attention 权重装进推理服务常驻表（LRU），
返回管道键给 catvton-tryon 复用。首次装载约 160s（3.4GB 基座反序列化），
之后命中 LRU 直接返回；`model-unload` 节点可按 `catvton:<variant>:<dtype>:...`
别名卸载。fp32 是 Pascal(sm_61) 的默认口径（非张量 fp16 吞吐 = fp32 的 1/64）。
"""
from executor_base import run_op
from flowx_client import param


def main():
    run_op("vton.catvton_load", {
        "variant": param("variant", "mix"),
        "dtype": param("dtype", "fp32"),
        "vae_dtype": param("vae_dtype", "fp32"),
        "chunk": int(param("chunk", "1024")),
    }, emit_keys=["vton", "info"], timeout=1800)


if __name__ == "__main__":
    main()
