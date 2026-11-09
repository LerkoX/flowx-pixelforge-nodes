"""video_mask 族：SAM3 视频人物遮罩（文本提示检测 + 全片追踪）。

- video.person_mask(video, prompt, obj_pick, dilate, max_frames)
  → mask VIDEO（灰度帧：255=被选中人物，0=背景）+ fps/帧数/宽高

用途：H3 人物替换（minimax.h3.gen 的 inpaint_mask）等多人精确编辑场景。
模型：facebook/sam3（transformers 集成 Sam3VideoModel + Sam3VideoProcessor，
检测器+追踪器一体），权重软链 compshare 共享盘
/model/ModelScope/facebook/sam3（config.json 从 ModelScope 补齐）。

输入 video 对象兼容两种 data 形态：{"path": ...}（load-video 直通，ffmpeg
解码）与 {"frames": [...], "fps": ...}（归一化对象，直接用帧）。
"""
import os
import time

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "48899affaaeb"
OP_FAMILY = "video_mask"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------


SAM3_DIR = os.environ.get(
    "SAM3_VIDEO_DIR", "/var/lib/flowx-inference/models/sam3-video")


def _models():
    from app.main import models
    return models


def _load_sam3():
    """SAM3 视频分割模型常驻（0.86B bf16，加载一次 ~5s）。"""
    key = "sam3-video:bf16"
    models = _models()
    if key in models._pipes:
        return models.get(key)
    import torch
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    t0 = time.time()
    model = Sam3VideoModel.from_pretrained(
        SAM3_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()
    processor = Sam3VideoProcessor.from_pretrained(SAM3_DIR)
    obj = {"model": model, "processor": processor}
    # est：3.45GB 权重（bf16 驻卡 ~1.8GB；记账按文件口径，与其他模型一致）
    est = sum(os.path.getsize(os.path.join(SAM3_DIR, f))
              for f in ("model.safetensors",)
              if os.path.isfile(os.path.join(SAM3_DIR, f)))
    with models._lock:
        models._evict_if_needed(est)
        models._pipes[key] = obj
        models._archs[key] = "sam3-video"
        models._sizes[key] = est
        models._last_used[key] = time.time()
    print(f"[video-mask] sam3-video loaded {time.time()-t0:.1f}s "
          f"est={est/1e9:.1f}GB", flush=True)
    return models.get(key)


def _video_to_frames(video_data, max_frames=0):
    """VIDEO data → (frames PIL 列表, fps)。path 形态走 ffmpeg 裸流解码。"""
    if isinstance(video_data, dict) and video_data.get("frames"):
        frames = video_data["frames"]
        if max_frames:
            frames = frames[:int(max_frames)]
        return frames, float(video_data.get("fps") or 24.0)
    path = (video_data or {}).get("path")
    if not path or not os.path.isfile(path):
        raise ValueError(f"video.person_mask: 视频对象无 frames 且 path 无效: "
                         f"{str(video_data)[:120]}")
    import subprocess
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate",
         "-of", "csv=p=0", path],
        capture_output=True, text=True, timeout=60)
    if probe.returncode != 0 or not probe.stdout.strip():
        raise ValueError(f"ffprobe 解析失败: {path}: {probe.stderr[:200]}")
    w_s, h_s, rate_s = probe.stdout.strip().split(",")
    width, height = int(w_s), int(h_s)
    num, _, den = rate_s.partition("/")
    fps = float(num) / float(den) if den and float(den) else float(num)
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_frames:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, timeout=600)
    if raw.returncode != 0:
        raise ValueError(f"ffmpeg 解码失败: {raw.stderr[:200]}")
    from PIL import Image
    fb = width * height * 3
    n = len(raw.stdout) // fb
    if n == 0:
        raise ValueError(f"视频无有效帧: {path}")
    frames = [Image.frombytes("RGB", (width, height),
                              raw.stdout[i*fb:(i+1)*fb]) for i in range(n)]
    return frames, fps


def _dilate_masks(masks, px):
    """(T, H, W) bool → maxpool 膨胀（无边距核近似，代替 cv2.dilate）。"""
    if px <= 0:
        return masks
    import torch
    x = torch.from_numpy(masks).float().unsqueeze(1)  # (T,1,H,W)
    k = int(px) * 2 + 1
    x = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=1,
                                       padding=int(px))
    return x.squeeze(1).numpy() > 0.5


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "video.person_mask",
        inputs={"video": "VIDEO", "prompt": "STRING", "obj_pick": "STRING",
                "dilate": "INT", "max_frames": "INT"},
        outputs={"mask": "VIDEO", "fps": "INT", "num_frames": "INT",
                 "width": "INT", "height": "INT", "objects": "INT"},
        description="SAM3 视频人物遮罩：文本提示（如 'the woman in white'）"
                    "检测目标并全片追踪，输出逐帧灰度遮罩（255=选中）。"
                    "obj_pick: all=该提示命中的全部目标合并 / best=仅最高分")
    def _op_person_mask(video, prompt="person", obj_pick="all", dilate=12,
                        max_frames=0):
        import numpy as np
        import torch
        from PIL import Image

        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("video.person_mask: prompt 不能为空")
        data = video  # registry 已解包到对象 data 本体
        frames, fps = _video_to_frames(data, int(max_frames or 0))
        n = len(frames)
        w, h = frames[0].size
        print(f"[video-mask] {n} 帧 {w}x{h} fps={fps:.2f} prompt={prompt!r}",
              flush=True)

        bundle = _load_sam3()
        model, processor = bundle["model"], bundle["processor"]

        t0 = time.time()
        session = processor.init_video_session(
            video=frames, inference_device="cuda",
            dtype=torch.bfloat16)
        session = processor.add_text_prompt(session, prompt)

        merged = np.zeros((n, h, w), dtype=bool)
        objects_seen = 0
        with torch.no_grad():
            for i in range(n):
                out = model(inference_session=session, frame_idx=i)
                post = processor.postprocess_outputs(session, out)
                masks = post["masks"]       # (num_obj, H, W) bool
                scores = post["scores"]     # (num_obj,)
                if masks.shape[0] == 0:
                    continue
                objects_seen = max(objects_seen, int(masks.shape[0]))
                m_np = masks.cpu().numpy()
                if str(obj_pick).lower() == "best" and m_np.shape[0] > 1:
                    top = int(scores.argmax().item())
                    m_np = m_np[top:top+1]
                fm = m_np.any(axis=0)
                # postprocess 输出是原分辨率 bool
                if fm.shape != (h, w):
                    fm_img = Image.fromarray(fm.astype(np.uint8) * 255)
                    fm = np.asarray(
                        fm_img.resize((w, h), Image.NEAREST)) > 127
                merged[i] |= fm
                if i % 20 == 0 or i == n - 1:
                    cover = float(merged[i].mean()) * 100
                    print(f"[video-mask] frame {i+1}/{n} "
                          f"obj={masks.shape[0]} cover={cover:.1f}%",
                          flush=True)

        merged = _dilate_masks(merged, int(dilate))
        empty = int((~merged.any(axis=(1, 2))).sum())
        if empty:
            print(f"[video-mask] 警告：{empty}/{n} 帧无遮罩（目标未检出/跟丢）",
                  flush=True)
        mask_frames = [Image.fromarray(merged[i].astype(np.uint8) * 255)
                       for i in range(n)]
        print(f"[video-mask] done {time.time()-t0:.1f}s "
              f"objects_peak={objects_seen}", flush=True)
        return {"mask": {"frames": mask_frames, "fps": fps},
                "fps": int(round(fps)), "num_frames": n,
                "width": w, "height": h, "objects": objects_seen}
