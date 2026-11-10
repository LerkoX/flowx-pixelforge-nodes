"""wan-animate-preprocess：Wan-Animate 输入预处理（参考视频 → 骨骼动作视频）。

## 格式实证（读 ComfyUI comfy_extras/nodes_wan.py 源码确认，2026-11-11）

**pose_video 没有私有骨骼格式要求**——ComfyUI 的 WanAnimateToVideo 拿到的就是
任意 RGB 帧批次，直接：

    pose_video = common_upscale(pose_video[:length], width, height, "area")
    pose_video_latent = vae.encode(pose_video[:, :, :, :3])
    positive = conditioning_set_values(positive, {"pose_video_latent": ...})

即骨骼可视化图（DWPose / OpenPose 渲染）走标准 VAE encode 变成 conditioning。
「预处理后的骨骼表示」= 这类可视化骨骼视频/帧序列，**不是二进制私有格式**。
（同一函数里 face_video 缩放到 512×512 归一化 [-1,1]；负向用全 -1。）

所以本节点用 **DWPose 133 关键点 + OpenPose 渲染** 产出的骨架视频完全对路。

## 实现

纯编排，不新增服务端算子：
  video.load → video.to_frames → preprocess.openpose(逐帧) → video.encode
（preprocess.py 已内联 DWPose ONNX：yolox_l 人检 + dw-ll_ucoco_384 关键点）

## face_video

ComfyUI v1 语义是人脸裁剪序列（内部缩放到 512×512）；Wan-Animate **2**
（WanAnimate2ToVideo）已**移除 face_video**，改用 pose_strength /
pose_start_percent / pose_end_percent / reference_image_strength 四个旋钮。
故 face_video 用原视频占位影响有限，pass_through_face 开关默认开。
"""

import time

from flowx_client import (emit, param, ref, submit_job,
                          wait_job, token)


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
        raise RuntimeError(
            "wan-animate-preprocess: video（原始参考视频对象 ID）必填")

    detector = param("detector", "dwpose")
    include_hand = param("include_hand", "0") in ("1", "true", "True")
    include_face = param("include_face", "0") in ("1", "true", "True")
    max_frames = param("max_frames", 0, int)
    stride = param("stride", 1, int)
    timeout = param("job_timeout", 3600, int)
    t0 = time.time()

    # ① 抽帧
    tf = {"name": op_tf,
          "inputs": {"video": ref(video_id), "stride": stride}}
    if max_frames > 0:
        tf["inputs"]["end"] = max_frames - 1
    out = _run(url, tok, tf, timeout)
    frames_id = out["frames"]
    fps = out.get("fps") or 16
    count = out.get("count") or 0
    print(f"[wan-animate-preprocess] frames={frames_id} count={count} "
          f"fps={fps} ({time.time()-t0:.1f}s)", flush=True)

    # ② 逐帧骨架提取（preprocess.openpose 收 IMAGE 批次，服务端内部遍历）
    pp = {"name": op_pp,
          "inputs": {"image": ref(frames_id), "detector": detector,
                     "include_hand": bool(include_hand),
                     "include_face": bool(include_face)}}
    out = _run(url, tok, pp, timeout)
    pose_frames_id = out["image"]
    print(f"[wan-animate-preprocess] pose frames={pose_frames_id} "
          f"detector={detector} ({time.time()-t0:.1f}s)", flush=True)

    # ③ 编码回视频（保持原 fps）
    enc = {"name": op_enc,
           "inputs": {"frames": ref(pose_frames_id), "fps": int(fps)}}
    out = _run(url, tok, enc, timeout)
    pose_video_id = out["video"]
    print(f"[wan-animate-preprocess] pose_video={pose_video_id} "
          f"({time.time()-t0:.1f}s)", flush=True)

    # ④ face_video：pass-through 原视频（占位；见模块 docstring）
    face_video_id = ""
    if param("pass_through_face", "1") in ("1", "true", "True"):
        face_video_id = video_id
        print("[wan-animate-preprocess] face_video = 原视频（pass-through "
              "占位，官方 face 特征格式未公开）", flush=True)

    emit(pose_video=pose_video_id, face_video=face_video_id,
         num_frames=count, fps=int(fps))


if __name__ == "__main__":
    main()
