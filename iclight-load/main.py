from executor_base import run_op
from flowx_client import param


def main():
    run_op("iclight.load", {
        "variant": param("variant", "fc"),
        "dtype": param("dtype", "auto"),
        "weights": param("weights", ""),
        "reload": param("reload", "false"),
    }, emit_keys=["model", "params_mb", "vram_mb", "info"], timeout=3600)


if __name__ == "__main__":
    main()
