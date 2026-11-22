"""video_face 族：视频脸部追踪裁切 + 修复回贴（PixelAI H3 FaceFix 路线）。

复刻 ComfyUI 社区 PixelAI Labs《H3 FaceFix》工作流的核心环节：
「H3 会把远景脸画熔化。修法不是重渲整段，而是把脸抠出来放大重画再贴回」。

- video.face_crop(video, mask, crop_factor, out_size, ema)
  遮罩 → 逐帧脸部包围盒（EMA 平滑防抖）→ 方形裁切放大到 out_size
  → 输出脸部特写视频 + 全白遮罩视频（接 minimax.h3.gen 的
  inpaint_video/inpaint_mask，整裁片重绘语义）+ boxes JSON + 脸部参考图
  （取脸部最大帧的裁片，作 ref2va 参考图）
- video.face_stitch(video, crop_video, boxes, feather, ...)
  把重绘后的脸部特写按 boxes 贴回原视频（高斯羽化边缘 + 可选脸部大小
  渐隐：远脸全修、近脸不动，对齐 Face Size Fade 节点语义），音轨透传。

与整帧 inpaint 的本质区别：裁片空间重画让脸吃到 out_size（768）的完整
分辨率——1344 高的竖屏视频里脸可能只有 100~200px，整帧重绘细节提升
有限；裁片重绘后贴回，脸部细节 = 768px 级渲染。
"""
import json
import os
import subprocess
import time

# --- F1 族版本化（与其他族同规）：算子身份 = 实现内容 ------------------------
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
OP_VERSION = "cac55ac70f7b"
OP_FAMILY = "video_face"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------


def _video_to_frames(video_data, max_frames=0):
    """VIDEO data → (frames PIL 列表, fps)。path 形态走 ffmpeg 裸流解码。"""
    if isinstance(video_data, dict) and video_data.get("frames"):
        frames = video_data["frames"]
        if max_frames:
            frames = frames[:int(max_frames)]
        return frames, float(video_data.get("fps") or 24.0)
    path = (video_data or {}).get("path")
    if not path or not os.path.isfile(path):
        raise ValueError(f"video_face: 视频对象无 frames 且 path 无效: "
                         f"{str(video_data)[:120]}")
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
                              raw.stdout[i * fb:(i + 1) * fb]) for i in range(n)]
    return frames, fps


def _mask_bbox(mask_img):
    """单帧灰度遮罩 → **最大连通白区**的包围盒 (x, y, w, h)；空遮罩返回 None。

    ⚠ 必须用最大连通域而非全局 bbox：SAM3 遮罩偶发伪影（实测一条
    1~2px 宽、贯穿全帧高的竖线），全局 bbox 会被它冲爆（face_h
    116→913）→ 裁窗瞬间变全身 → 该帧重绘整个场景 → 贴回后闪动/偏色
    （1997 根因）。cv2 不可用时回退全局 bbox（旧行为）。
    """
    import numpy as np
    m = np.asarray(mask_img.convert("L")) > 127
    if not m.any():
        return None
    try:
        import cv2
        n_lab, _lab, stats, _cent = cv2.connectedComponentsWithStats(
            m.astype(np.uint8), connectivity=8)
        if n_lab > 1:
            # stats[:, 4] = area；跳过背景（label 0）取最大
            k = 1 + int(np.argmax(stats[1:, 4]))
            x, y, w, h = (int(v) for v in stats[k, :4])
            return (x, y, w, h)
    except ImportError:
        pass
    rows = np.where(m.any(axis=1))[0]
    cols = np.where(m.any(axis=0))[0]
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    return (x0, y0, x1 - x0, y1 - y0)


def _extract_audio(video_data):
    """从源视频对象提取音轨 → (audio ndarray (2,N) float32, sample_rate)。

    对象已带 audio 字段直接透传；path 形态用 ffmpeg 解码成 32kHz 立体声
    f32（与对象仓库 encode_mp4 的口径一致）。失败返回 (None, None)——
    音轨是尽力而为，不影响主流程。
    """
    if isinstance(video_data, dict) and video_data.get("audio") is not None:
        return video_data["audio"], int(video_data.get("sample_rate") or 32000)
    path = (video_data or {}).get("path")
    if not path or not os.path.isfile(path):
        return None, None
    try:
        import numpy as np
        sr = 32000
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-vn",
             "-ac", "2", "-ar", str(sr), "-f", "f32le", "-"],
            capture_output=True, timeout=120)
        if raw.returncode != 0 or not raw.stdout:
            return None, None
        pcm = np.frombuffer(raw.stdout, dtype=np.float32)
        if pcm.size % 2:
            pcm = pcm[:-1]
        audio = pcm.reshape(-1, 2).T.copy()  # (2, N)
        return audio, sr
    except Exception as e:  # noqa: BLE001
        print(f"[video-face] 音轨提取失败（忽略）: {e}", flush=True)
        return None, None


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "video.face_crop",
        inputs={"video": "VIDEO", "mask": "VIDEO",
                "crop_factor": "FLOAT", "out_size": "INT", "ema": "FLOAT"},
        outputs={"crop": "VIDEO", "mask_white": "VIDEO", "boxes": "STRING",
                 "face_ref": "IMAGE", "num_frames": "INT", "fps": "INT",
                 "face_h_max": "INT"},
        description="脸部追踪裁切（PixelAI FaceFix 第一段）：遮罩逐帧出脸部"
                    "包围盒（最大连通白区），EMA 平滑防抖，按 crop_factor "
                    "（脸高倍数，社区默认 2.5）取方形盒裁切并放大到 "
                    "out_size×out_size。输出：脸部特写视频（接 H3Gen "
                    "inpaint_video）+ 全白遮罩视频（接 inpaint_mask，整裁片"
                    "重绘语义，靠 soft 线性升温锚姿态）+ boxes JSON（接 "
                    "video.face_stitch）+ 脸部参考图（脸部最大帧的裁片，"
                    "可作 ref2va 参考图）")
    def _op_face_crop(video, mask, crop_factor=2.5, out_size=768, ema=0.75):
        import numpy as np
        from PIL import Image

        t0 = time.time()
        frames, fps = _video_to_frames(video, 0)
        mframes, _ = _video_to_frames(mask, 0)
        n = len(frames)
        if len(mframes) != n:
            raise ValueError(
                f"video.face_crop: 遮罩帧数 {len(mframes)} != 视频帧数 {n}")
        W, H = frames[0].size
        cf = max(1.2, float(crop_factor))
        out = max(64, int(out_size))
        ema_f = min(0.99, max(0.0, float(ema)))

        # 第一遍：收集原始包围盒（不做任何平滑）
        raw = []  # (cx, cy, w, h) 或 None
        for i in range(n):
            bb = _mask_bbox(mframes[i])
            if bb is None:
                raw.append(None)
            else:
                x, y, w, h = bb
                raw.append((x + w / 2.0, y + h / 2.0, float(w), float(h)))
        if raw[0] is None and not any(r is not None for r in raw):
            raise ValueError(
                "video.face_crop: 全程检不到脸（遮罩全空）——"
                "检查上游遮罩（prompt 是否命中脸部）")

        # **尺寸离群值剔除**（1997 根因修复）：SAM3 的 face 遮罩会偶发
        # 吞掉整个人（实测 face_h 116→913 尖峰），EMA 根本拦不住，裁窗
        # 瞬间变成全身 → 该帧重绘整个场景 → 贴回后紫色调+内容突变=闪动。
        # 对齐官方 AutoContext 的轨迹仲裁思路：rolling 中位数窗口，
        # 偏离 [0.55×, 1.8×] 的帧判为检测失败 → 走缺口插值。
        hs = np.array([r[3] if r else np.nan for r in raw])
        valid = ~np.isnan(hs)
        win = 9
        for i in range(n):
            if not valid[i]:
                continue
            lo, hi = max(0, i - win // 2), min(n, i + win // 2 + 1)
            med = np.nanmedian(hs[lo:hi][valid[lo:hi]]) \
                if valid[lo:hi].any() else np.nan
            if np.isnan(med):
                continue
            if hs[i] < 0.55 * med or hs[i] > 1.8 * med:
                raw[i] = None  # 离群 → 判失败，交给缺口插值
                valid[i] = False
        n_outlier = int((~valid).sum())

        # 缺口插值（含丢帧）：最近有效帧之间线性插值 cx/cy/w/h；
        # 首尾缺口用最近有效帧填充
        vidx = np.where(valid)[0]
        if len(vidx) == 0:
            raise ValueError("video.face_crop: 离群剔除后无有效帧")
        interp = []
        for i in range(n):
            if valid[i]:
                interp.append(raw[i])
                continue
            j = np.searchsorted(vidx, i)
            if j == 0:
                interp.append(raw[vidx[0]])
            elif j >= len(vidx):
                interp.append(raw[vidx[-1]])
            else:
                a, b = vidx[j - 1], vidx[j]
                t = (i - a) / float(b - a)
                interp.append(tuple(
                    raw[a][k] * (1 - t) + raw[b][k] * t for k in range(4)))

        # 第二遍：EMA 平滑（位置+尺寸同源序列）
        boxes = []
        raw_face_h = []
        prev = None
        for i in range(n):
            cx, cy, w, h = interp[i]
            raw_face_h.append(int(round(h)))
            side = max(w, h) * cf
            if prev is not None:
                cx = ema_f * prev[0] + (1 - ema_f) * cx
                cy = ema_f * prev[1] + (1 - ema_f) * cy
                side = ema_f * prev[2] + (1 - ema_f) * side
            prev = (cx, cy, side)
            # 方形盒 clamp 到帧内
            side = min(side, W, H)
            x0 = int(round(min(max(cx - side / 2, 0), W - side)))
            y0 = int(round(min(max(cy - side / 2, 0), H - side)))
            boxes.append({"x": x0, "y": y0, "w": int(round(side)),
                          "h": int(round(side)),
                          "face_h": int(raw_face_h[-1])})

        face_h_max = max(raw_face_h)
        # 裁切 + 放大
        crops = []
        for i in range(n):
            b = boxes[i]
            c = frames[i].crop((b["x"], b["y"], b["x"] + b["w"],
                                b["y"] + b["h"])).resize(
                                    (out, out), Image.LANCZOS)
            crops.append(c)
        # 全白遮罩（整裁片重绘语义：mask=1 全编辑区，源锚定靠 soft 调度）
        white = Image.new("RGB", (out, out), (255, 255, 255))
        mask_white = [white.copy() for _ in range(n)]
        # 脸部参考图：脸部最大帧的裁片
        ref_idx = int(np.argmax(raw_face_h))
        face_ref = crops[ref_idx]
        print(f"[video-face] crop done {time.time()-t0:.1f}s: {n} 帧 "
              f"{W}x{H} → {out}x{out} | 脸高 {min(raw_face_h)}~{face_h_max}px "
              f"| 离群剔除 {n_outlier} 帧 | 参考帧={ref_idx} "
              f"boxes[0]={boxes[0]}", flush=True)
        import base64
        boxes_json = json.dumps(boxes)
        # Studio 字符串模板引擎会把合法 JSON 文本当结构化值重新渲染
        # （实测输出 '<[]interface {} Value>'）——boxes 一律 base64 传输，
        # stitch 端自动解码（兼容原始 JSON）。
        boxes_b64 = base64.b64encode(boxes_json.encode()).decode()
        return {"crop": {"frames": crops, "fps": fps},
                "mask_white": {"frames": mask_white, "fps": fps},
                "boxes": boxes_b64,
                "face_ref": face_ref,
                "num_frames": n, "fps": int(round(fps)),
                "face_h_max": int(face_h_max)}

    @registry.register(
        "video.face_stitch",
        inputs={"video": "VIDEO", "crop_video": "VIDEO", "boxes": "STRING",
                "feather": "INT", "full_fix_below": "INT",
                "no_fix_above": "INT", "color_match": "INT"},
        outputs={"video": "VIDEO", "fps": "INT", "num_frames": "INT"},
        description="脸部修复回贴（PixelAI FaceFix 末段）：把重绘后的脸部"
                    "特写按 boxes 逐帧缩放贴回原视频，高斯羽化 feather px "
                    "消边缘。脸部大小渐隐（对齐 Face Size Fade）：face_h ≤ "
                    "full_fix_below 全量修复，≥ no_fix_above 保持原样（近景"
                    "脸不动，不多添皱纹），中间线性过渡；两者为 0 关闭。"
                    "音轨从源视频透传。")
    def _op_face_stitch(video, crop_video, boxes, feather=24,
                        full_fix_below=0, no_fix_above=0, color_match=1):
        import base64
        from PIL import Image, ImageFilter

        t0 = time.time()
        frames, fps = _video_to_frames(video, 0)
        crops, _ = _video_to_frames(crop_video, 0)
        n = len(frames)
        if len(crops) != n:
            raise ValueError(
                f"video.face_stitch: 裁片帧数 {len(crops)} != 源视频帧数 {n}")
        boxes_s = boxes if isinstance(boxes, str) else boxes.decode()
        try:
            boxes_l = json.loads(boxes_s)
        except Exception:  # noqa: BLE001 — base64 传输形态（见 face_crop）
            boxes_l = json.loads(base64.b64decode(boxes_s).decode())
        if len(boxes_l) != n:
            raise ValueError(
                f"video.face_stitch: boxes 数 {len(boxes_l)} != 帧数 {n}")
        feather = max(0, int(feather))
        ffb, nfa = int(full_fix_below), int(no_fix_above)

        def _ring_stats(arr, ring_frac=0.25):
            """外环区域的逐通道 mean/std（patch 边缘应与源背景衔接的颜色口径）。"""
            import numpy as np
            h, w = arr.shape[:2]
            rw, rh = max(2, int(w * ring_frac)), max(2, int(h * ring_frac))
            ring = np.zeros((h, w), dtype=bool)
            ring[:rw, :] = ring[-rw:, :] = True
            ring[:, :rh] = ring[:, -rh:] = True
            sel = arr[ring]  # (K,3)
            return sel.mean(axis=0), sel.std(axis=0)

        out_frames = []
        # **颜色匹配（治蓝色方块闪动 + 帧级色偏）**：H3 重绘的裁片与源帧
        # 存在色温/曝光漂移。统计量逐帧收集（中位数抗手臂入侵污染），
        # 再对 gain/bias 序列做 **时序 EMA 平滑**——折中方案：
        # - 纯逐帧（第一版）：每帧增益不同 → 色温闪动（1991 踩坑）
        # - 纯全局常量（第二版）：治不了生成侧的帧级色偏（运动模糊帧
        #   整片泛紫，1996 踩坑）
        # - EMA 平滑（本版）：跟随慢速光照变化 + 修正帧级色偏，
        #   平滑掉帧间跳变不引入新闪动
        cm_series = None
        if int(color_match):
            import numpy as np
            per_frame = []  # (p_m, s_m, gain) per frame
            for i in range(n):
                b = boxes_l[i]
                src_np = np.asarray(
                    frames[i].convert("RGB").crop(
                        (b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"])),
                    dtype=np.float32)
                pat_np = np.asarray(
                    crops[i].convert("RGB").resize(
                        (b["w"], b["h"]), Image.LANCZOS), dtype=np.float32)
                s_m, s_s = _ring_stats(src_np)
                p_m, p_s = _ring_stats(pat_np)
                gain = np.where(p_s > 1e-3, s_s / np.maximum(p_s, 1e-3), 1.0)
                gain = np.clip(gain, 0.5, 2.0)  # 防爆
                per_frame.append((p_m, s_m, gain))
            # 时序 EMA（alpha=0.3：响应 ~5 帧时间尺度，快于光照变化、
            # 慢于单帧抖动）；首尾用前向/后向填充
            alpha = 0.3
            smoothed = []
            acc = per_frame[0]
            for pf in per_frame:
                acc = tuple(alpha * cur + (1 - alpha) * prev
                            for cur, prev in zip(pf, acc))
                smoothed.append(acc)
            cm_series = smoothed
            g = np.stack([s[2] for s in smoothed])
            print(f"[video-face] color_match(EMA): gain范围 "
                  f"[{g.min(axis=0).round(2)},{g.max(axis=0).round(2)}]",
                  flush=True)
        for i in range(n):
            b = boxes_l[i]
            base = frames[i].convert("RGB")
            patch = crops[i].convert("RGB").resize((b["w"], b["h"]),
                                                   Image.LANCZOS)
            if cm_series is not None:
                import numpy as np
                p_m, s_m, gain = cm_series[i]
                pat_np = np.asarray(patch, dtype=np.float32)
                pat_np = (pat_np - p_m) * gain + s_m
                patch = Image.fromarray(
                    np.clip(pat_np, 0, 255).astype(np.uint8))
            # 羽化 alpha：盒内全白，边缘高斯过渡
            alpha = Image.new("L", (b["w"], b["h"]), 255)
            if feather > 0:
                # 先收缩 feather/2 再模糊，保证盒中心区域纯白
                inner = Image.new("L", (b["w"], b["h"]), 0)
                from PIL import ImageDraw
                d = ImageDraw.Draw(inner)
                f2 = feather // 2
                d.rectangle([f2, f2, b["w"] - f2 - 1, b["h"] - f2 - 1],
                            fill=255)
                alpha = inner.filter(
                    ImageFilter.GaussianBlur(radius=max(1, feather / 2)))
            # Face Size Fade：远脸全修 / 近脸不动 / 中间线性
            scale = 1.0
            if ffb > 0 and nfa > ffb:
                fh = float(b.get("face_h") or 0)
                if fh >= nfa:
                    scale = 0.0
                elif fh > ffb:
                    scale = (nfa - fh) / float(nfa - ffb)
            if scale <= 0.0:
                out_frames.append(base)
                continue
            if scale < 1.0:
                alpha = alpha.point(lambda v: int(v * scale))
            base.paste(patch, (b["x"], b["y"]), alpha)
            out_frames.append(base)

        audio, sr = _extract_audio(video)
        result = {"frames": out_frames, "fps": fps}
        if audio is not None:
            result["audio"] = audio
            result["sample_rate"] = sr
        print(f"[video-face] stitch done {time.time()-t0:.1f}s: {n} 帧 "
              f"feather={feather} fade=({ffb},{nfa}) audio="
              f"{'yes' if audio is not None else 'no'}", flush=True)
        return {"video": result, "fps": int(round(fps)), "num_frames": n}
