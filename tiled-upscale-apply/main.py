from executor_base import run_op
from flowx_client import param, ref


def main():
    ups = param("upscale_model", "")
    inputs = {
        "image": ref(param("image")),
        "model": ref(param("model")),
        "pos": ref(param("pos")),
        "neg": ref(param("neg")),
        "plan": param("plan", ""),
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
        "cfg": float(param("cfg", "7.0")),
        "sampler_name": param("sampler_name", ""),
        "scheduler": param("scheduler", "normal"),
    }
    if ups:
        inputs["upscale_model"] = ref(ups)
    run_op("tiled.apply", inputs,
           emit_keys=["image", "pre_upscaled", "seam_map", "tiles", "coverage",
                      "width", "height"], timeout=3600)


if __name__ == "__main__":
    main()
