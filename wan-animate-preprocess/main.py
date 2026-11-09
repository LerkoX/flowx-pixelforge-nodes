"""wan-animate-preprocess：Wan-Animate 输入预处理（参考视频 → 骨骼动作视频）。

⚠ 重要说明（先读）：
Wan-Animate 官方预处理脚本**未公开代码**（humanaigc/Wan-Animate 只有项目
页），diffusers 也只说「见官方仓库」。本节点采用论文描述的最接近实现：
用 **DWPose（133 关键点 wholebody）逐帧提取 + OpenPose 风格渲染**产出
`pose_video`——即 Wan-Animate 期望的 "spatially-aligned skeleton signals"。

若实测与官方权重不匹配，两条退路：
1. 用官方脚本离线产出 pose/face 视频，再用 load-video 上传，直接喂
   wan-animate（本节点产出的对象 ID 与 load-video 等价）
2. 调整 detector（openpose / dwpose）与渲染参数再试

实现方式：**纯编排**，不新增算子——
  video.load → video.to_frames → preprocess.openpose(逐帧) →
  video.encode
全部复用现成算子，无新增服务端代码。

face_video：官方为「implicit facial features」，公开资料不足以可靠复现。
本节点提供 pass_through_face 开关：把原视频直接作为 face_video 占位
（部分权重容忍），或留空让 wan-animate 只用 pose。实测后再决定是否需要
专门的 face 提取算子。
"""
import time

from flowx_client import (emit, ensure_plugin, param, ref, submit_job,
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

    ensure_plugin(url, "video.to_frames", tok=tok)
    ensure_plugin(url, "preprocess.openpose", tok=tok)
    ensure_plugin(url, "video.encode", tok=tok)

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
    tf = {"name": "video.to_frames",
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
    pp = {"name": "preprocess.openpose",
          "inputs": {"image": ref(frames_id), "detector": detector,
                     "include_hand": bool(include_hand),
                     "include_face": bool(include_face)}}
    out = _run(url, tok, pp, timeout)
    pose_frames_id = out["image"]
    print(f"[wan-animate-preprocess] pose frames={pose_frames_id} "
          f"detector={detector} ({time.time()-t0:.1f}s)", flush=True)

    # ③ 编码回视频（保持原 fps）
    enc = {"name": "video.encode",
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
