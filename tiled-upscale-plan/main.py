from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("tiled.plan", {
        "image": ref(param("image")),
        "scale": float(param("scale", "2.0")),
        "target_w": int(param("target_w", "0")),
        "target_h": int(param("target_h", "0")),
        "tile": int(param("tile", "768")),
        "overlap": int(param("overlap", "96")),
        "pre": param("pre", "lanczos"),
        "denoise": float(param("denoise", "0.35")),
        "seed": int(param("seed", "-1")),
        "seam_fix": param("seam_fix", "mask"),
        "band_denoise": float(param("band_denoise", "0.25")),
        "steps": int(param("steps", "20")),
    }, emit_keys=["plan", "grid_image", "tiles", "rows", "cols", "target_w",
                  "target_h", "est_samples", "target_size"])


if __name__ == "__main__":
    main()
