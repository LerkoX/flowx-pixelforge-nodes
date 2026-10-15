"""video-repaint：视频到视频（逐帧重绘）—— 换人 / 转绘 / 风格化的主力节点。

每帧 VAE 编码 → 以 strength 为 denoise 重采样 → 解码（服务端 video.repaint 算子）：
帧间保持原运动（不需要运动模块），pos 接 IPAdapter / InstantID / Redux 的 COND
即可把参考人物/风格逐帧注入 ⇒ 跳舞视频换人；control_net 可选按帧注入结构条件，
在更高 strength 下保住姿态。

为什么逐帧而不是整段 v2v：逐帧路径对底模无要求、显存占用与单帧 img2img 同级，
且每帧解码结果都能立刻上画布（preview_every）；整段 v2v（video-v2v 节点）时间
一致性更好但需要 v2v 管道。seed 只随机一次后逐帧继承，抑制帧间闪烁。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    inputs = {
        "model": ref(param("model")),
        "pos": ref(param("pos")),
        "neg": ref(param("neg")),
        "video": ref(param("video")),
        "strength": param("strength", 0.6, float),
        "steps": param("steps", 20, int),
        "cfg": param("cfg", 7.0, float),
        "seed": param("seed", -1, int),
        "sampler_name": param("sampler_name", ""),
        "scheduler": param("scheduler", ""),
        "start_at_step": param("start_at_step", 0, int),
        "end_at_step": param("end_at_step", 0, int),
        "start": param("start", 0, int),
        "end": param("end", 0, int),
        "stride": param("stride", 1, int),
        "max_frames": param("max_frames", 0, int),
        "control_strength": param("control_strength", 1.0, float),
        "preview_every": param("preview_every", 1, int),
    }
    # 可选对象输入：空串表示不用（不传端口 = 算子默认 None，E3 口径不受影响）
    control_net = param("control_net", "")
    if control_net:
        inputs["control_net"] = ref(control_net)
    control_frames = param("control_frames", "")
    if control_frames:
        inputs["control_frames"] = ref(control_frames)

    run_op("video.repaint", inputs,
           emit_keys=["video", "seed", "num_frames"], timeout=7200)


if __name__ == "__main__":
    main()
