from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("colorize.apply", {
        "image": ref(param("image")),
        "model": param("model"),
        "size": int(param("size", "512")),
        "saturation": float(param("saturation", "1.0")),
    }, emit_keys=["image", "width", "height", "chroma", "chroma_in", "info"],
        timeout=1800, input_media=("image",))


if __name__ == "__main__":
    main()
