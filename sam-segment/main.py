from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("seg.sam", {
        "image": ref(param("image")),
        "model": param("model", "sam:vit_l"),
        "points": param("points", ""),
        "box": param("box", ""),
        "multimask": param("multimask", "true"),
        "pick": int(param("pick", "0")),
        "overlay": float(param("overlay", "0.5")),
    }, emit_keys=["mask", "candidates", "score", "annotated", "width", "height"])


if __name__ == "__main__":
    main()
