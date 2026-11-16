"""video-enhance：视频二次修复（服务端插件 video_enhance.py）。

逐帧 GFPGAN/RestoreFormer 面部修复（只改人脸，羽化贴回）+ RealESRGAN/SwinIR
模型超分；两阶段独立开关。权重在推理服务 /models/facerestore 与 /models/upscale。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("video.enhance", {
        "video": ref(param("video")),
        "face_model": param("face_model", "gfpgan"),
        "upscale_model": param("upscale_model", ""),
        "det_thresh": float(param("det_thresh", "0.5")),
        "face_padding": float(param("face_padding", "0.3")),
        "face_feather": int(param("face_feather", "16")),
        "face_weight": float(param("face_weight", "0.5")),
        "tile": int(param("tile", "0")),
        "fps": int(param("fps", "0")),
    }, emit_keys=["video", "faces", "scale"])


if __name__ == "__main__":
    main()
