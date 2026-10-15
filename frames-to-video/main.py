"""frames-to-video：帧序列（IMAGE 批量对象）→ VIDEO（编码 mp4）。

与 video-to-frames 成对：帧序列进出都用同一对象 ID，帧本体始终留在服务端对象
仓库（不经过 FlowX 控制通道）。编码由对象仓库 put 时完成（imageio-ffmpeg）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("video.encode", {
        "frames": ref(param("frames")),
        "fps": param("fps", 24, int),
    }, emit_keys=["video", "num_frames"])


if __name__ == "__main__":
    main()
