from executor_base import run_op
from flowx_client import param, ref


def main():
    args = {
        "image": ref(param("image")),
        "model": param("model"),
        "light_dir": param("light_dir", "none"),
        "prompt": param("prompt", ""),
        "negative_prompt": param("negative_prompt", ""),
        "a_prompt": param("a_prompt", "best quality"),
        "width": int(param("width", "0")),
        "height": int(param("height", "0")),
        "steps": int(param("steps", "25")),
        "cfg": float(param("cfg", "2.0")),
        "seed": int(param("seed", "12345")),
        "denoise_low": float(param("denoise_low", "0.9")),
        "denoise_high": float(param("denoise_high", "0.5")),
        "highres_scale": float(param("highres_scale", "1.5")),
        "highres": param("highres", "true"),
        "scheduler": param("scheduler", "dpmpp"),
    }
    # 可选图像：留空则不传（服务端取默认 None）
    mask_id = param("mask", "")
    if mask_id:
        args["mask"] = ref(mask_id)
    bg_id = param("bg", "")
    if bg_id:
        args["bg"] = ref(bg_id)
    run_op("iclight.relight", args,
           emit_keys=["image", "width", "height", "light_balance", "lum_left",
                      "lum_right", "info"], timeout=3600, input_media=("image",))


if __name__ == "__main__":
    main()
