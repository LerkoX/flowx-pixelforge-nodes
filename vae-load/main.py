"""vae-load：外挂 VAE 加载（vae.load）。

外挂 VAE（vae-ft-mse 等）的意义是解码质量；dtype=auto 默认 fp32。
输出 VAE 句柄直接喂 vae-decode / vae-encode 替代管道内置 VAE。
"""
from executor_base import run_op
from flowx_client import param


def main():
    run_op("vae.load", {
        "name": param("name"),
        "dtype": param("dtype", "auto"),
    }, emit_keys=["vae"])


if __name__ == "__main__":
    main()
