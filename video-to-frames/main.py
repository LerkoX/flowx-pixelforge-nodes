"""video-to-frames：VIDEO → 帧序列（IMAGE 批量对象）。

抽帧口径与 load-video 一致（start/end/stride/max_frames）；帧序列是**单个 IMAGE
对象的批量形态**（多帧共用一个对象 ID，GET /images/{id}?index=n 取第 n 帧），
避免几十个对象 ID 在节点间传递。下游 frames-to-video 收同一对象回编码成视频。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("video.to_frames", {
        "video": ref(param("video")),
        "start": param("start", 0, int),
        "end": param("end", 0, int),
        "stride": param("stride", 1, int),
        "max_frames": param("max_frames", 0, int),
    }, emit_keys=["frames", "fps", "count"])


if __name__ == "__main__":
    main()
