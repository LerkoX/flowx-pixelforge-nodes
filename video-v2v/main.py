"""video-v2v：整段视频原生 v2v 重采样（AnimateDiff 等运动模型的 v2v 管道）。

与 video-repaint（逐帧）的分工：本节点保留原片的时间/运动结构（运动模块在整段
latent 上建模），一致性好、闪烁少；但对管道有要求——必须接受 `video` 参数
（AnimateDiffVideoToVideoPipeline / Wan/LTX 系 v2v）。算子会先查管道签名，
不支持即报错，**不会**退化成 t2v 出无关视频。

输入帧序列由服务端从源片解码（源片先用 load-video 登记成 VIDEO 对象）。
"""
from executor_base import run_op
from flowx_client import param, ref


def _pbool(raw):
    """布尔参数解析（env 注入是字符串，bool("false") 为真是历史陷阱）。"""
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def main():
    run_op("video.sample_v2v", {
        "model": ref(param("model")),
        "video": ref(param("video")),
        "prompt": param("prompt", ""),
        "neg_prompt": param("neg_prompt", ""),
        "strength": param("strength", 0.6, float),
        "steps": param("steps", 25, int),
        "cfg": param("cfg", 7.5, float),
        "seed": param("seed", -1, int),
        "width": param("width", 0, int),
        "height": param("height", 0, int),
        "num_frames": param("num_frames", 0, int),
        "fps": param("fps", 0, int),
        "decode_chunk_size": param("decode_chunk_size", 0, int),
        "preview_every": param("preview_every", 5, int),
        # False（默认）= diffusers 原生截断语义（有效步数 = int(steps×strength)）；
        # True = ComfyUI 语义（steps 即实际步数，strength 只缩放 sigma 排布）
        "enforce_inference_steps": _pbool(param("enforce_inference_steps", "false")),
    }, emit_keys=["video", "seed"], timeout=7200)


if __name__ == "__main__":
    main()
