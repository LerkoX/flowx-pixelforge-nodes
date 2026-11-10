"""video-control-preprocess：视频 → 逐帧控制信号视频（depth / pose / canny）。

**为什么需要它**（真机教训）：
Wan-VACE 的 `video` 参数如果直接喂 **RGB 原视频**，是「软条件」——模型要自己
从像素猜结构，帧间猜法不一致就会出现**画面上下跳转 / 后期崩坏成土灰色**
（真机复现）。ComfyUI 的 Wan-Animate 用 **pose 骨架**做
"spatially-aligned skeleton signals"，即**显式结构信号**，模型不用猜。

本节点把原视频逐帧转成 depth/pose/canny 控制视频，喂给 wan-vace 的 video 端口
（或 wan-animate 的 pose_video），让结构由控制信号决定、外观由 reference_image 决定。

纯编排，不新增服务端算子：
  video.load → video.to_frames → preprocess.<mode>(逐帧) → video.encode
"""
import time

from flowx_client import (emit, param, ref, resolve_op, submit_job,
                          wait_job, token)

_OPS = {"depth": "preprocess.depth", "pose": "preprocess.openpose",
        "canny": "preprocess.canny"}


def _run(url, tok, payload, timeout):
    jid = submit_job(url, payload, tok)
    res = wait_job(url, jid, tok, timeout=timeout,
                   poll=param("poll_interval", 5, int))
    return {k: m.get("id", m.get("value"))
            for k, m in res.get("outputs", {}).items()}


def main():
    url = param("service_url").rstrip("/")
    tok = token()


    video_id = param("video", "")
    if not video_id:
        raise RuntimeError("video-control-preprocess: video 必填")

    mode = (param("mode", "depth") or "depth").lower()
    if mode not in _OPS:
        raise RuntimeError(f"mode 需为 {list(_OPS)}，got {mode}")
    # F1 之后算子名带版本戳，纯编排节点用 resolve_op 查实际名（不自注册）
    op_tf = resolve_op(url, "video.to_frames", tok)
    op_enc = resolve_op(url, "video.encode", tok)
    op_name = resolve_op(url, _OPS[mode], tok)

    t0 = time.time()
    timeout = param("job_timeout", 3600, int)

    # ① 抽帧
    tf = {"name": op_tf,
          "inputs": {"video": ref(video_id),
                     "stride": param("stride", 1, int)}}
    mf = param("max_frames", 0, int)
    if mf > 0:
        tf["inputs"]["end"] = mf - 1
    out = _run(url, tok, tf, timeout)
    frames_id, fps, count = out["frames"], out.get("fps") or 16, out.get("count") or 0

    print(f"[vcpp] frames={count} fps={fps} ({time.time()-t0:.1f}s)", flush=True)

    # ② 逐帧预处理 → 控制信号帧
    # ⚠ preprocess.depth/openpose/canny 只接受**单个 IMAGE**（真机：
    # "image 需为 IMAGE 对象，got list"），而 video.to_frames 给的是帧列表。
    # 故这里逐帧调用（串行；81 帧约 2~3 分钟），把结果帧 id 收集成新列表。
    # preprocess 族已支持批量（list→list），一次调用处理整批帧
    if mode == "depth":
        inputs = {"image": ref(frames_id),
                  "detect_resolution": param("detect_resolution", 512, int)}
    elif mode == "pose":
        inputs = {"image": ref(frames_id),
                  "detector": param("detector", "dwpose"),
                  "include_hand": param("include_hand", "0") in ("1", "true", "True"),
                  "include_face": param("include_face", "0") in ("1", "true", "True")}
    else:
        inputs = {"image": ref(frames_id),
                  "low_threshold": param("low_threshold", 100, int),
                  "high_threshold": param("high_threshold", 200, int)}
    out = _run(url, tok, {"name": op_name, "inputs": inputs}, timeout)
    ctrl_frames = out["image"]
    print(f"[vcpp] {mode} 控制帧 {ctrl_frames} ({time.time()-t0:.1f}s)", flush=True)

    # ③ 编码回视频（保持原 fps）
    out = _run(url, tok, {"name": op_enc,
                          "inputs": {"frames": ref(ctrl_frames), "fps": int(fps)}},
               timeout)
    print(f"[vcpp] control_video={out['video']} ({time.time()-t0:.1f}s)",
          flush=True)
    emit(video=out["video"], num_frames=count, fps=int(fps))


if __name__ == "__main__":
    main()
