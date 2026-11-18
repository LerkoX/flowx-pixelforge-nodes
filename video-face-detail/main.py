"""video-face-detail：视频级 SD 修脸（服务端插件 sd_face.py）。

逐帧 YOLO 检测 → 裁剪 → SD 低 denoise img2img → 羽化贴回；
检测框 EMA 时序平滑 + 固定 seed 抑制帧间跳变；可选 ControlNet tile 锁轮廓。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("video.face_detail", {
        "video": ref(param("video")),
        "ckpt": param("ckpt", "Realistic_Vision_V5.1"),
        "prompt": param("prompt", ""),
        "negative": param("negative", ""),
        "detector": param("detector", "face"),
        "det_conf": float(param("det_conf", "0.5")),
        "padding": float(param("padding", "0.4")),
        "denoise": float(param("denoise", "0.35")),
        "steps": int(param("steps", "20")),
        "cfg": float(param("cfg", "6.0")),
        "guide_size": int(param("guide_size", "512")),
        "feather": int(param("feather", "16")),
        "max_targets": int(param("max_targets", "1")),
        "seed": int(param("seed", "20261116")),
        "sampler_name": param("sampler_name", "euler"),
        "scheduler": param("scheduler", "normal"),
        "controlnet": param("controlnet", ""),
        "cn_strength": float(param("cn_strength", "0.6")),
        "temporal_alpha": float(param("temporal_alpha", "0.6")),
        "fps": int(param("fps", "0")),
    }, emit_keys=["video", "faces"])


if __name__ == "__main__":
    main()
