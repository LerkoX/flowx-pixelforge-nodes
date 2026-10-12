from executor_base import run_op
from flowx_client import param


def main():
    run_op("colorize.load", {
        "variant": param("variant", "modelscope"),
        "dtype": param("dtype", "fp32"),
        "reload": param("reload", "false"),
    }, emit_keys=["model", "params_mb", "vram_mb", "info"], timeout=1800)


if __name__ == "__main__":
    main()
