from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("seg.grounding_detect", {
        "image": ref(param("image")),
        "model": param("model", "gd:tiny"),
        "prompt": param("prompt", "person"),
        "box_threshold": float(param("box_threshold", "0.35")),
        "text_threshold": float(param("text_threshold", "0.25")),
        "max_boxes": int(param("max_boxes", "10")),
        "sam_model": param("sam_model", ""),
        "overlay": float(param("overlay", "0.5")),
    }, emit_keys=["annotated", "boxes", "labels", "scores", "count", "mask",
                  "width", "height"])


if __name__ == "__main__":
    main()
