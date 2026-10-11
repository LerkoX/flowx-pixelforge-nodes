from executor_base import run_op
from flowx_client import param


def main():
    run_op("seg.sam_load", {
        "variant": param("variant", "vit_l"),
        "dtype": param("dtype", "fp32"),
    }, emit_keys=["model", "params_mb", "vram_mb"], timeout=1200)


if __name__ == "__main__":
    main()
