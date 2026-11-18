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
OP_VERSION = "e9433f65a9d0"
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
# **SAM2（2026-11-17）**：换 SAM2 视频追踪解决 SAM3 的 mask 精度问题。
# SAM3 的 mask 实际分辨率只有原图 1/8（768x1344 → 96x168）→ 放大后是
# 8px 方块阶梯；SAM2 的 mask 网格是 256x256，bilinear 上采样后边缘平滑。
# 组合用法：SAM3 只负责**第一帧出框**（它的文本检测很强），SAM2 负责全片
# 追踪（它的记忆库时序稳定 + mask 更细）。
SAM2_DIR = os.environ.get(
    "SAM2_VIDEO_DIR", "/var/lib/flowx-inference/models/sam2")

def _load_sam2():
    """SAM2 视频模型常驻（0.081B bf16，加载 ~1s）。"""
    key = "sam2-video:bf16"
    models = _models()
    if key in models._pipes:
        return models.get(key)
    import torch
    from transformers import Sam2VideoModel, Sam2VideoProcessor
    t0 = time.time()
    model = Sam2VideoModel.from_pretrained(
        SAM2_DIR, torch_dtype=torch.bfloat16).to("cuda").eval()
    processor = Sam2VideoProcessor.from_pretrained(SAM2_DIR)
    obj = {"model": model, "processor": processor}
    est = sum(os.path.getsize(os.path.join(SAM2_DIR, f))
              for f in ("model.safetensors",)
              if os.path.isfile(os.path.join(SAM2_DIR, f)))
    with models._lock:
        models._evict_if_needed(est)
        models._pipes[key] = obj
        models._archs[key] = "sam2-video"
        models._sizes[key] = est
        models._last_used[key] = time.time()
    print(f"[video-mask] sam2-video loaded {time.time()-t0:.1f}s "
          f"est={est/1e9:.1f}GB", flush=True)
    return models.get(key)

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

def _motion_union(masks, frames, thresh=25, near_px=30):
    """帧差运动区 ∪ 分割 mask（薄纱裙摆等高速运动区域的分割漏网兜底）。

    根因（成片 248 复盘）：SAM3 对快速运动的半透明薄纱裙摆追踪失败，
    mask 只罩躯干 → 裙摆区被 bg 抹黑 → diffusers replace 无语义合成
    （mask 仅作条件信号）→ 模型忠实保留黑色区 = 黑色裙摆残留。
    帧差与分割模型无关，运动必捕获。邻域约束防止远处运动物（窗帘摆动等）
    被误吞：只保留与人物 mask 膨胀 near_px 邻域相交的运动区。
    """
    import numpy as np
    import cv2
    T, H, W = masks.shape
    out = masks.copy()
    n_add = 0
    for t in range(T):
        t0 = max(0, t - 1)
        t1 = min(T - 1, t + 1)
        a = cv2.cvtColor(np.asarray(frames[t0]), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(np.asarray(frames[t1]), cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(a, b)
        motion = diff > int(thresh)
        if not motion.any():
            continue
        # 邻域约束：运动区必须接触（膨胀后的）人物 mask
        near = cv2.dilate(masks[t].astype(np.uint8),
                          np.ones((int(near_px) * 2 + 1,) * 2, np.uint8)) > 0
        extra = motion & near & ~masks[t]
        if extra.any():
            # 运动区本身毛糙：闭运算填洞 + 轻微膨胀连成整片
            e = extra.astype(np.uint8)
            e = cv2.morphologyEx(e, cv2.MORPH_CLOSE,
                                 np.ones((9, 9), np.uint8))
            e = cv2.dilate(e, np.ones((5, 5), np.uint8))
            # **边缘平滑链（tapered_corners 等效，2026-11-15 裙摆暗带修复）**：
            # 帧差补丁的方形角/毛刺边缘会让模型在粗糙边界生成暗过渡带。
            # 开运算去毛刺 → 大椭圆核闭运算平滑拐角 → 小高斯软化边缘。
            e = cv2.morphologyEx(e, cv2.MORPH_OPEN,
                                 np.ones((7, 7), np.uint8))
            e = cv2.morphologyEx(e, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(
                                     cv2.MORPH_ELLIPSE, (31, 31)))
            e = cv2.GaussianBlur(e, (0, 0), 2.0)
            e = (e > 0.5).astype(np.uint8)
            out[t] |= e > 0
            n_add += int((e > 0).sum())
    print(f"[video-mask] 帧差运动兜底: 补 {n_add} px", flush=True)
    return out

def _bbox_rectify(masks, smooth=5):
    """(T,H,W) bool → **矩形包围盒 mask**（Wan2.2 官方 get_aug_mask
    w_len=1/h_len=1 等价物）+ bbox 跨帧滑动平均防抖 + 空帧近邻填补。

    官方 replace 配方用矩形而非精细轮廓（arxiv 2502.06145）：模型在矩形
    黑洞里生成人物连同身边环境，边缘过渡远离人体轮廓 → 天然无轮廓黑影、
    光照一致。精细轮廓 mask 逼迫模型贴轮廓画人 → 轮廓黑影+图层感。
    """
    import numpy as np
    T, H, W = masks.shape
    boxes = np.zeros((T, 4), dtype=np.float32)
    valid = np.zeros(T, dtype=bool)
    for t in range(T):
        ys, xs = np.nonzero(masks[t])
        if len(xs):
            boxes[t] = [xs.min(), ys.min(), xs.max(), ys.max()]
            valid[t] = True
    if not valid.any():
        return masks  # 全空：原样返回（上层有告警）
    # 空帧近邻填补（前向优先）
    for t in range(T):
        if not valid[t]:
            for d in range(1, T):
                if t - d >= 0 and valid[t - d]:
                    boxes[t] = boxes[t - d]
                    break
                if t + d < T and valid[t + d]:
                    boxes[t] = boxes[t + d]
                    break
    # bbox 坐标滑动平均防抖（矩形逐帧跳动会产生边缘闪烁）
    if int(smooth) > 1:
        k = np.ones(int(smooth)) / int(smooth)
        for c in range(4):
            boxes[:, c] = np.convolve(boxes[:, c], k, mode="same")
    out = np.zeros_like(masks)
    for t in range(T):
        x0, y0, x1, y1 = boxes[t].astype(int)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W - 1, x1), min(H - 1, y1)
        if x1 > x0 and y1 > y0:
            out[t, y0:y1 + 1, x0:x1 + 1] = True
    return out

def _guided_filter(guide, src, r=8, eps=1e-4):
    """原图引导的 mask 精修（He et al. guided filter，boxFilter 实现，
    不依赖 opencv-contrib 的 ximgproc）。

    **为什么必须做（2026-11-16 真机）**：SAM3 的 mask 实际分辨率只有原图的
    1/8（`obj_id_to_mask` 是 low-res，`postprocess_outputs` 的 original_sizes
    会被 session 的 video_height/width 抢先，改不到），放大回原尺寸就是**8px
    方块阶梯**——用户反馈"蒙版没精确捕捉到人"、"依旧不平滑"。
    引导滤波用**原视频灰度**做 guide：平坦区强平滑、边缘处保边，能把阶梯
    边界吸附到真实人物轮廓上（细节保留远好于单纯 blur / morphology）。
    """
    import numpy as np
    import cv2 as _cv2

    I = guide.astype(np.float32) / 255.0
    p = src.astype(np.float32)
    k = (2 * r + 1, 2 * r + 1)
    mean_I = _cv2.boxFilter(I, -1, k)
    mean_p = _cv2.boxFilter(p, -1, k)
    corr_I = _cv2.boxFilter(I * I, -1, k)
    corr_Ip = _cv2.boxFilter(I * p, -1, k)
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = _cv2.boxFilter(a, -1, k)
    mean_b = _cv2.boxFilter(b, -1, k)
    return mean_a * I + mean_b

def _dilate_masks(masks, px):
    """(T, H, W) bool 膨胀。

    **改用椭圆核 cv2.dilate（2026-11-17）**：原实现是 maxpool2d（方核）——
    方核膨胀会把平滑的 mask 边界推成**方块/阶梯**，这正是「换了 SAM2 之后
    轮廓依然不平滑」的原因。椭圆核按半径扩展，边界保持圆弧过渡。
    """
    if px <= 0:
        return masks
    import cv2 as _cv2
    import numpy as _np
    k = _cv2.getStructuringElement(_cv2.MORPH_ELLIPSE,
                                   (int(px) * 2 + 1,) * 2)
    out = _np.empty_like(masks, dtype=bool)
    for i in range(masks.shape[0]):
        out[i] = _cv2.dilate(masks[i].astype(_np.uint8), k) > 0
    return out

DEPTH_DIR = os.environ.get(
    "DEPTH_ANYTHING_DIR", "/var/lib/flowx-inference/models/depth-anything-v2")


def _load_depth_anything():
    """Depth-Anything-V2-Base 常驻（0.097B bf16）。

    **HF 版本键名映射（2026-11-17 真机）**：transformers 5.x 的 DINOv2 实现
    用 `attention.q_proj/k_proj/v_proj/o_proj`，而 HF Hub 上 depth-anything-v2
    的权重还是旧命名 `attention.attention.query/key/value` + `attention.output.dense`
    → 直接 from_pretrained 会 96 MISSING / 96 UNEXPECTED（主干随机初始化，
    深度图完全不可用）。这里手动 load + 映射表，实测 0/0。
    """
    key = "depth-anything-v2:bf16"
    models = _models()
    if key in models._pipes:
        return models.get(key)
    import json as _json

    import torch
    from safetensors.torch import load_file
    from transformers import AutoImageProcessor, DepthAnythingConfig,         DepthAnythingForDepthEstimation
    t0 = time.time()
    cfg = DepthAnythingConfig(
        **_json.load(open(os.path.join(DEPTH_DIR, "config.json"))))
    m = DepthAnythingForDepthEstimation(cfg).to(torch.bfloat16)
    sd = load_file(os.path.join(DEPTH_DIR, "model.safetensors"))
    _MAP = {".attention.attention.query.": ".attention.q_proj.",
            ".attention.attention.key.": ".attention.k_proj.",
            ".attention.attention.value.": ".attention.v_proj.",
            ".attention.output.dense.": ".attention.o_proj."}
    new = {}
    for k, v in sd.items():
        for a, b in _MAP.items():
            if a in k:
                k = k.replace(a, b)
        new[k] = v
    miss, _ = m.load_state_dict(new, strict=False)
    del sd, new
    m = m.to("cuda").eval()
    proc = AutoImageProcessor.from_pretrained(DEPTH_DIR)
    obj = {"model": m, "processor": proc}
    est = os.path.getsize(os.path.join(DEPTH_DIR, "model.safetensors")) \
        if os.path.isfile(os.path.join(DEPTH_DIR, "model.safetensors")) else 0
    with models._lock:
        models._evict_if_needed(est)
        models._pipes[key] = obj
        models._archs[key] = "depth-anything-v2"
        models._sizes[key] = est
        models._last_used[key] = time.time()
    print(f"[video-mask] depth-anything loaded {time.time()-t0:.1f}s "
          f"missing={len(miss)}", flush=True)
    return models.get(key)


def _load_depth_video_frames(depth_video):
    """把 video-depth 输出的深度 VIDEO 解成 [(H,W) float 0~1 近亮]。"""
    import numpy as np

    frames, _fps = _video_to_frames(depth_video, 0)
    out = []
    for fr in frames:
        a = np.asarray(fr.convert("L"), dtype=np.float32) / 255.0
        out.append(a)
    return out


def _trimap_matting(masks, frames, depth_maps=None, px=8, guide_r=6):
    """Trimap 抠图 → 软 alpha（T,H,W float 0~1）。

    参考 Comfy-Pro-Alpha-Generator 的 CPA Final 思路：**只在 SAM 输出不确定的
    区域（边界带）用深度做前后判断**，确定前景/背景完全保留，因此不会破坏
    人体内部或吃掉背景。

      1) 确定前景 core = erode(mask, px)       —— 必属人物
      2) 确定背景 bg   = ~dilate(mask, px)     —— 必属背景
      3) 未知带 band   = 其余（mask 边界的锯齿/抖动区）
      4) band 内 alpha 用**深度的线性解**：把人物 core 的平均深度记作 1、
         背景平均深度记作 0，逐像素按深度插值 → 深度在边缘处比颜色可靠
         （用户实测：人物 0.573±0.084 vs 背景 0.130）
      5) 再用**原图灰度做引导滤波**把 alpha 吸附到真实轮廓（发丝/手指级）
    输出是**连续 alpha**（不是 0/255 二值），下游 composite 直接当软 mask 用，
    可消除"硬边 + 黑描边"。
    """
    import cv2
    import numpy as np

    h, w = masks.shape[1], masks.shape[2]
    # **核心区腐蚀要小**（2026-11-17 真机，关键）：原用 px*2+1（px=10 → 21px）
    # 做 core 腐蚀，但手臂宽度往往才 15~25px → 整条手臂被划进"未知带"，深度
    # 在细长区域估计不稳 → 手臂内部被误判为背景 → 成片手臂上出现深色斑块。
    # 改为 core=腐蚀 px/2（5px）：手臂内部仍是"确定前景"，未知带只覆盖真正的
    # 边缘 ±10px（dilate px*2 范围内），既保留软边又不误伤细小肢体。
    k_core = max(3, ((px // 2) * 2 + 1) | 1)
    k_out = max(5, (px * 2 + 1) | 1)
    out = np.zeros((len(masks), h, w), dtype=np.float32)
    n_band = 0
    for i in range(len(masks)):
        m = masks[i].astype(np.uint8)
        if not m.any():
            continue
        core = cv2.erode(m, np.ones((k_core, k_core), np.uint8)) > 0
        bg = ~(cv2.dilate(m, np.ones((k_out, k_out), np.uint8)) > 0)
        band = ~(core | bg)
        n_band += int(band.sum())
        alpha = np.zeros((h, w), dtype=np.float32)
        alpha[core] = 1.0
        if band.any():
            if depth_maps is not None and i < len(depth_maps):
                d = depth_maps[i]
                if d.shape != (h, w):
                    d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
            else:
                # 无外部深度：退化用"到 core 的距离场"做 alpha（也能出软边）
                d = cv2.distanceTransform(
                    (core.astype(np.uint8)), cv2.DIST_L2, 5)
            d_core = float(d[core].mean()) if core.any() else float(d.max())
            d_bg = float(d[bg].mean()) if bg.any() else float(d.min())
            span = d_core - d_bg if abs(d_core - d_bg) > 1e-6 else None
            # **保守策略：深度只做"减法"（2026-11-17 真机）**
            # 原实现按深度线性插值给 alpha（d→0 背景、d→1 前景），但深度在
            # **细长肢体**（手臂/手指）上估计不可靠——手臂与背景深度接近时
            # 会把手臂边缘判成背景，alpha<1 → 合成混入背景 → 成片手臂上出现
            # 深色斑块（实测两次复现）。
            # 改为：未知带内**默认按 mask 保留（alpha=1）**，只有当像素深度
            # **明显接近背景**（< 背景均值 + 0.15*跨度）时才判为背景剔除。
            # 即只削掉"突出到背景里的那圈"，绝不主动把肢体内部降为半透明。
            if span is None:
                a = np.ones_like(d, dtype=np.float32)
            else:
                thr_bg = d_bg + 0.15 * abs(span)
                # 深度越大越近（Depth-Anything 原生）；保守只剔除"很像背景"的
                a = np.where(d <= thr_bg, 0.0, 1.0).astype(np.float32)
            # 引导滤波：用原图灰度做 guide，把 alpha 贴到真实轮廓
            try:
                gray = np.asarray(frames[i].convert("L"), dtype=np.float32)
                a = _guided_filter(gray, a.astype(np.float32),
                                   r=guide_r, eps=1e-4)
            except Exception:  # noqa: BLE001
                pass
            alpha[band] = np.clip(a[band], 0.0, 1.0)
        out[i] = alpha
    print(f"[video-mask] trimap 抠图 px={px} guide_r={guide_r} "
          f"未知带累计 {n_band} 像素", flush=True)
    return out


def _estimate_depth_frames(frames, invert=False):
    """逐帧 Depth-Anything-V2 深度估计 → [(H,W) float 0~1]（近=大）。"""
    import numpy as np
    import torch

    b = _load_depth_anything()
    m, proc = b["model"], b["processor"]
    h, w = frames[0].size[1], frames[0].size[0]
    out = []
    with torch.no_grad():
        for i, fr in enumerate(frames):
            inp = proc(images=fr, return_tensors="pt").to("cuda")
            d = m(**inp).predicted_depth
            d = d if d.ndim == 2 else d[0]
            d = torch.nn.functional.interpolate(
                d[None, None], size=(h, w), mode="bilinear",
                align_corners=False)[0, 0].float().cpu().numpy()
            dn = (d - d.min()) / (d.max() - d.min() + 1e-6)
            out.append(1.0 - dn if invert else dn)
            if i % 20 == 0 or i == len(frames) - 1:
                print(f"[video-depth] {i+1}/{len(frames)}", flush=True)
    return out


def _smooth_contour(mask, win=15):
    """**轮廓级平滑**（2026-11-17，治 mask 边缘凹凸不平）。

    为什么像素级形态学治不了：用户的抱怨是"边缘很多凹凸"——那是**轮廓上的
    高频抖动**（SAM2 的 256 网格双线性上采样锯齿 + dilate 的局部突出 +
    depth_refine 的硬阈值判定 → 方块凸起）。开闭运算只能磨圆角，无法
    消除轮廓路径本身的抖动。

    做法（经典 mask 精修）：
      1. `findContours` 取最大外轮廓（无损取点 CHAIN_APPROX_NONE）
      2. 对轮廓点做**循环滑动平均**（首尾相接，避免接缝）—— 等价于在轮廓
         参数化空间做低通，直接抹掉高频抖动
      3. `fillPoly` 重新栅格化 → 得到光滑、无凸起的 mask
    内部空洞一并按同样方式处理（保留手指缝等大空洞，只平滑其轮廓）。
    """
    import numpy as np
    import cv2 as _cv2

    m = (mask.astype(np.uint8))
    n_lbl, lbl = _cv2.connectedComponents(m, connectivity=8)
    if n_lbl <= 1:
        return mask
    out = np.zeros_like(m)
    k = max(1, int(win))
    for li in range(1, n_lbl):
        comp = (lbl == li).astype(np.uint8)
        if comp.sum() < 200:          # 噪点直接丢
            continue
        cnts, _ = _cv2.findContours(comp, _cv2.RETR_CCOMP,
                                    _cv2.CHAIN_APPROX_NONE)
        filled = np.zeros_like(comp)
        for c in cnts:
            pts = c.reshape(-1, 2).astype(np.float64)
            n = len(pts)
            if n < 2 * k + 3:
                _cv2.fillPoly(filled, [pts.astype(np.int32)], 1)
                continue
            pad = np.vstack([pts[-k:], pts, pts[:k]])
            ker = np.ones(2 * k + 1) / (2 * k + 1)
            sm = np.stack([np.convolve(pad[:, 0], ker, "same")[k:k + n],
                           np.convolve(pad[:, 1], ker, "same")[k:k + n]],
                          axis=1)
            _cv2.fillPoly(filled, [sm.round().astype(np.int32)], 1)
        out |= filled
    return out.astype(bool)


def _depth_refine(masks, frames, px=12, margin=0.05, depth_maps=None):
    """深度辅助 mask 精修（2026-11-17）。

    **为什么需要**：SAM2/SAM3 的轮廓与人物真实边界总有偏差，超出的部分在
    替换后被模型画成暗色 ⇒ 黑边/黑雾。深度图给出了**空间前后关系**——人物
    通常明显比背景近（实测人物 0.573±0.084 vs 背景 0.130），这个区分度远高于
    颜色/纹理，正好用来在**边界带内**判定像素归属：
      1) core  = erode(mask, 较深)  → 确信的人物内部，取深度均值作参照
      2) band  = dilate(mask, px) - erode(mask, 较小) → 待判定环带
      3) 带内像素深度 > (人物均值 - margin) 的**归入人物**，其余**剔除**
    """
    import cv2
    import numpy as np
    import torch
    from PIL import Image as _Image

    # **外部深度优先（2026-11-17）**：若上游传了 video-depth 的产物，
    # 直接消费它的灰度帧（省一次推理，且深度可单独预览/复用）；否则内部
    # 加载 Depth-Anything-V2 自己估。
    h, w = masks.shape[1], masks.shape[2]
    _ext = None
    if depth_maps is not None and len(depth_maps):
        _ext = []
        for _d in depth_maps:
            # 兼容两种来源：video-depth 解的 **numpy 数组**（0~1）与 PIL 帧
            if hasattr(_d, "convert"):
                _a = np.asarray(_d.convert("L"), dtype=np.float32) / 255.0
                if _a.shape != (h, w):
                    _a = np.asarray(_d.convert("L").resize((w, h)),
                                    dtype=np.float32) / 255.0
            else:
                _a = np.asarray(_d, dtype=np.float32)
                if _a.max() > 1.5:
                    _a = _a / 255.0
                if _a.shape != (h, w):
                    import cv2 as _cvd
                    _a = _cvd.resize(_a, (w, h),
                                     interpolation=_cvd.INTER_LINEAR)
            _ext.append(_a)
        print(f"[video-mask] 用外部深度视频（{len(_ext)} 帧）", flush=True)
    if _ext is None:
        b = _load_depth_anything()
        m_d, proc = b["model"], b["processor"]
    kr_core = max(9, (px * 3) | 1)
    kr_in = max(3, (px // 2) | 1)
    kr_out = max(5, (px * 2 + 1) | 1)
    out = np.zeros_like(masks, dtype=bool)
    n_fix = 0
    with torch.no_grad():
        for i in range(len(frames)):
            if _ext is not None:
                dn = _ext[i] if i < len(_ext) else _ext[-1]
            else:
                inp = proc(images=frames[i], return_tensors="pt").to("cuda")
                d = m_d(**inp).predicted_depth
                d = d if d.ndim == 2 else d[0]
                d = torch.nn.functional.interpolate(
                    d[None, None], size=(h, w), mode="bilinear",
                    align_corners=False)[0, 0].float().cpu().numpy()
                dn = (d - d.min()) / (d.max() - d.min() + 1e-6)
            mk = masks[i].astype(np.uint8)
            core = cv2.erode(mk, np.ones((kr_core, kr_core), np.uint8)) > 0
            if not core.any():
                out[i] = masks[i]
                continue
            d_core = float(dn[core].mean())
            sd_core = float(dn[core].std()) or 0.08
            thr = d_core - max(margin, sd_core * 0.8)
            inner = cv2.erode(mk, np.ones((kr_in, kr_in), np.uint8)) > 0
            outer = cv2.dilate(mk, np.ones((kr_out, kr_out), np.uint8)) > 0
            band = outer & ~inner
            refined = inner | (band & (dn > thr))
            # 去掉零星孤岛（深度噪声），保留主体
            refined = cv2.morphologyEx(
                (refined.astype(np.uint8) * 255), cv2.MORPH_OPEN,
                np.ones((3, 3), np.uint8)) > 127
            n_fix += int(refined.sum() != masks[i].sum())
            out[i] = refined
            if i % 20 == 0 or i == len(frames) - 1:
                print(f"[video-mask] depth-refine {i+1}/{len(frames)} "
                      f"cover {masks[i].mean()*100:.1f}% -> "
                      f"{refined.mean()*100:.1f}%", flush=True)
    print(f"[video-mask] depth-refine 完成 px={px} 改动帧 {n_fix}/{len(frames)}",
          flush=True)
    return out


def _track_sam2(frames, prompt="person", obj_pick="all"):
    """SAM3 出首帧框 → SAM2 视频追踪 → (T,H,W) bool mask。

    为什么这样组合（2026-11-17 真机）：
      - SAM3 的文本检测很强（"person" 直接命中，无需外部检测器），但它的
        mask 输出只有原图 1/8 精度 → 8px 方块阶梯 → 合成后出黑边；
      - SAM2 的 mask 网格 256x256、有记忆库跨帧传播 → 边界平滑、时序稳定；
        但它需要**几何提示**（框/点），自己不会"找东西"。
      → 各取所长：SAM3 只跑**第一帧**拿最高分框，SAM2 拿这个框跑全片。
    """
    import numpy as np
    import torch

    h, w = frames[0].size[1], frames[0].size[0]
    b3 = _load_sam3()
    b2 = _load_sam2()

    # 1) SAM3 首帧检测 → 框
    s3 = b3["processor"].init_video_session(
        video=frames[:1], inference_device="cuda", dtype=torch.bfloat16)
    s3 = b3["processor"].add_text_prompt(s3, prompt)
    with torch.no_grad():
        o3 = b3["model"](inference_session=s3, frame_idx=0)
        post3 = b3["processor"].postprocess_outputs(
            s3, o3, original_sizes=[[h, w]])
    boxes = post3.get("boxes")
    scores = post3.get("scores")
    if boxes is None or len(boxes) == 0:
        raise ValueError(f"video.person_mask(sam2): SAM3 首帧未检出 "
                         f"prompt={prompt!r}")
    if str(obj_pick).lower() == "best" and len(boxes) > 1:
        bi = int(scores.argmax().item())
        box = boxes[bi].tolist()
    else:
        # all：取所有框的外接矩形（SAM2 单 obj 提示，多目标合并成一个）
        b = boxes.cpu().numpy()
        box = [float(b[:, 0].min()), float(b[:, 1].min()),
               float(b[:, 2].max()), float(b[:, 3].max())]
    print(f"[video-mask] SAM2 组合：SAM3 首帧框 "
          f"[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]", flush=True)

    # 2) SAM2 全片追踪
    s2 = b2["processor"].init_video_session(
        video=frames, inference_device="cuda", dtype=torch.bfloat16)
    _r = b2["processor"].add_inputs_to_inference_session(
        s2, frame_idx=0, obj_ids=[1], input_boxes=[[box]])
    if _r is not None:
        s2 = _r
    out = []
    with torch.no_grad():
        for i in range(len(frames)):
            o = b2["model"](inference_session=s2, frame_idx=i)
            # **SAM2 与 SAM3 的 API 差异**：每帧要把输出里的新 session 取回
            if getattr(o, "inference_session", None) is not None:
                s2 = o.inference_session
            pm = o.pred_masks.float()
            up = torch.nn.functional.interpolate(
                pm, size=(h, w), mode="bilinear", align_corners=False)
            # **保留 soft 概率**（不提前 >0 阈值化）：后面 guided filter
            # 需要连续场才能把边界吸附到原图真实边缘
            mk = torch.sigmoid(up[0, 0]).cpu().numpy()
            out.append(mk)
            if i % 20 == 0 or i == len(frames) - 1:
                print(f"[video-mask] sam2 frame {i+1}/{len(frames)} "
                      f"cover={(mk > 0.5).mean()*100:.1f}%", flush=True)
    return np.stack(out)  # float 概率 (T,H,W)



def _temporal_vote(masks, win=3):
    """时序多数表决：消除逐帧独立推理的阈值闪烁（0,1,0 → 0）。

    深度估计（DA-V2）逐帧独立 → 二值化后边界帧间跳动；RVM/SAM2 有记忆
    不需要，此 helper 主要给 depth 分支用。奇数窗，边界重复填充。
    """
    import numpy as np
    k = int(win) if int(win) % 2 == 1 else int(win) + 1
    half = k // 2
    m = np.asarray(masks, dtype=np.int16)
    n = len(m)
    pad = np.concatenate([np.repeat(m[:1], half, 0), m,
                          np.repeat(m[-1:], half, 0)], axis=0)
    votes = np.zeros_like(m)
    for off in range(k):
        votes += pad[off:off + n]
    return votes > half


def _depth_binary_masks(depths, min_area_ratio=0.15):
    """深度图 → 前景二值 mask：全片统一 Otsu 阈值 + 形态学清理 + 连通域过滤。

    为什么全片统一阈值：逐帧 Otsu 会随直方图变化漂移 → mask 边界闪烁；
    抽样帧聚合直方图做一次 Otsu，全片共用 → 时序稳定。
    连通域过滤：保留面积 ≥ 最大块 min_area_ratio 的域（人物分裂成两段
    躯干时不丢），剔除与人物无关的小前景碎块。
    """
    import cv2
    import numpy as np
    step = max(1, len(depths) // 16)
    allv = np.concatenate(
        [np.asarray(d, dtype=np.float32).ravel() for d in depths[::step]])
    allv = allv[:: max(1, len(allv) // 400000)]  # 采样上限 40 万点
    thr_u, _ = cv2.threshold((allv * 255).astype(np.uint8), 0, 255,
                             cv2.THRESH_OTSU)
    thr = float(thr_u) / 255.0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    out = []
    for d in depths:
        m = (np.asarray(d, dtype=np.float32) > thr).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
        n_lbl, lbl, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        if n_lbl > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = np.nonzero(areas >= areas.max() * min_area_ratio)[0]
            mm = np.zeros_like(m)
            for li in keep:
                mm |= (lbl == li + 1)
            m = mm
        out.append(m > 0)
    return np.stack(out), thr


def _matting_rvm(frames, weights=None, dsr=0.25):
    """RVM（Robust Video Matting）视频人物抠像：循环记忆保证时序稳定，
    输出**天然软 alpha**（0~1 连续，发丝级过渡），免 prompt、免深度辅助。

    - 权重是 torchscript（.pth 自带网络定义），torch.jit.load 即用；
    - 逐帧喂 (1,3,H,W)，rec 记忆跨帧保持；
    - ⚠ 抠的是**全部前景人物**（无 prompt 选择性），多人场景会把所有人
      都抠出来——需要指定某一人时用 sam2/sam3 分支。
    """
    import numpy as np
    import torch
    from pathlib import Path
    cands = []
    if weights:
        cands.append(str(weights))
    cands += [
        "/var/lib/flowx-inference/models/rvm/rvm_mobilenetv3.pth",
        "/var/lib/flowx-inference/models/rvm/mobilenetv3.pth",
        "/var/lib/flowx-inference/models/rvm/rvm_resnet50.pth",
        "/models/rvm/rvm_mobilenetv3.pth",
    ]
    path = next((c for c in cands if Path(c).is_file()), None)
    if path is None:
        raise RuntimeError(
            "RVM 权重未找到（rvm_mobilenetv3.pth 25MB / rvm_resnet50.pth "
            "100MB，github PeterL1n/RobustVideoMatting releases 或 hf-mirror）"
            "。请放到 /var/lib/flowx-inference/models/rvm/")
    model = torch.jit.load(path).cuda().eval()
    alphas = []
    rec = [None] * 4
    with torch.no_grad():
        for f in frames:
            src = torch.from_numpy(
                np.asarray(f.convert("RGB"), dtype=np.float32) / 255.0
            ).permute(2, 0, 1).unsqueeze(0).cuda()
            fgr, pha, *rec = model(src, *rec, downsample_ratio=float(dsr))
            alphas.append(pha[0, 0].float().cpu().numpy())
    return np.stack(alphas), path



_BIREF_CACHE = {}


def _matting_birefnet(frames, weights=None, smooth=3):
    """BiRefNet / RMBG-2.0 单帧高精度抠图（SOTA 边缘，双分辨率全局参考）。

    - **torch 直载**（共享盘 RMBG-2.0 目录自带自包含 birefnet.py 网络定义
      2244 行：PVT+Swin backbone 全内联，仅依赖 timm）+ safetensors 全量灌入；
    - RMBG-2.0 = BiRefNet 架构的**人像/前景特化版**（BRIA 官方免费非商用），
      人物场景首选；
    - 输入 1024² ImageNet 归一化，输出 sigmoid 前景 alpha；
    - **单帧无记忆** → 软 alpha 中心加权滑动平均时序平滑（无滞后）。
    """
    import cv2
    import numpy as np
    import torch
    from pathlib import Path
    cands = [str(weights)] if weights else []
    cands += [
        "/var/lib/flowx-inference/models/birefnet/RMBG-2.0",
        "/models/birefnet/RMBG-2.0",
        "/model/HuggingFace/chflame163/ComfyUI_LayerStyle/"
        "ComfyUI/models/BiRefNet/RMBG-2.0",
    ]
    rdir = next((c for c in cands
                 if c and (Path(c) / "birefnet.py").is_file()
                 and (Path(c) / "model.safetensors").is_file()), None)
    if rdir is None:
        raise RuntimeError(
            "BiRefNet/RMBG-2.0 模型目录未找到（需含 birefnet.py + "
            "model.safetensors；共享盘 /model/HuggingFace/chflame163/"
            "ComfyUI_LayerStyle/ComfyUI/models/BiRefNet/RMBG-2.0 可软链）")
    global _BIREF_CACHE
    if rdir not in _BIREF_CACHE:
        import importlib
        import sys
        import types
        # **手造包加载（2026-11-17）**：birefnet.py 内部用相对导入
        # （from .BiRefNet_config import ...），直接 sys.path 注入会
        # ImportError；把 RMBG-2.0 目录注册为包（rmbg2）再 importlib
        # 加载子模块，相对导入正常解析（且不污染 sys.path）。
        if "rmbg2" not in sys.modules:
            _pkg = types.ModuleType("rmbg2")
            _pkg.__path__ = [rdir]
            sys.modules["rmbg2"] = _pkg
        BiRefNet = importlib.import_module("rmbg2.birefnet").BiRefNet
        from safetensors.torch import load_file
        net = BiRefNet(bb_pretrained=False).cuda().eval()
        sd = load_file(str(Path(rdir) / "model.safetensors"))
        miss, unexp = net.load_state_dict(sd, strict=False)
        miss = [k for k in miss if "bb." not in k]
        if miss or len(unexp) > 0:
            print(f"[video-mask] birefnet 权重匹配 missing={len(miss)} "
                  f"unexpected={len(unexp)}", flush=True)
        _BIREF_CACHE[rdir] = net
        print(f"[video-mask] birefnet torch 加载: {rdir} "
              f"missing={len(miss)} unexpected={len(unexp)}", flush=True)
    net = _BIREF_CACHE[rdir]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    w0, h0 = frames[0].size
    alphas = []
    with torch.no_grad():
        for f in frames:
            img = cv2.resize(np.asarray(f.convert("RGB")), (1024, 1024),
                             interpolation=cv2.INTER_AREA)
            x = (img.astype(np.float32) / 255.0 - mean) / std
            x = torch.from_numpy(x.transpose(2, 0, 1))[None].cuda()
            preds = net(x)
            a = preds[-1].sigmoid()[0, 0].float().cpu().numpy()
            alphas.append(np.clip(
                cv2.resize(a, (w0, h0), interpolation=cv2.INTER_LINEAR),
                0, 1).astype(np.float32))
    A = np.stack(alphas)
    k = max(1, int(smooth)) | 1
    if k >= 3:
        half = k // 2
        pad = np.concatenate([np.repeat(A[:1], half, 0), A,
                              np.repeat(A[-1:], half, 0)], axis=0)
        ker = np.ones(k, dtype=np.float32)
        ker[half] = float(k)          # 中心帧权重最高（无滞后）
        ker /= ker.sum()
        A = np.stack([(pad[i:i + k] * ker[:, None, None]).sum(0)
                      for i in range(len(A))]).astype(np.float32)
    return A, rdir


def _emit_soft_mask(alpha, dilate, fps, n, w, h, tag, t0):
    """软 alpha mask 的公共收尾（rvm / birefnet 共用）：

    dilate>0 时做**软膨胀**（对连续 alpha 是 max 滤波，边缘向外渐开），
    然后直出 0~255 灰度帧——不做二值化/轮廓平滑（软 alpha 无锯齿语义）。
    """
    import numpy as np
    from PIL import Image
    if int(dilate) > 0:
        import cv2
        _k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (int(dilate) * 2 + 1, int(dilate) * 2 + 1))
        alpha = np.stack([
            cv2.dilate((np.clip(a, 0, 1) * 255).astype(np.uint8), _k
                       ).astype(np.float32) / 255.0 for a in alpha])
        print(f"[video-mask] {tag} 软膨胀 dilate={dilate}", flush=True)
    out_frames = [Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8))
                  for a in alpha]
    print(f"[video-mask] done({tag}) {time.time() - t0:.1f}s", flush=True)
    return {"mask": {"frames": out_frames, "fps": fps},
            "fps": int(round(fps)), "num_frames": n,
            "width": w, "height": h, "objects": 1}


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    @registry.register(
        "video.bg_erase",
        inputs={"video": "VIDEO", "mask": "VIDEO", "dilate": "INT",
                "radius": "INT", "method": "STRING"},
        outputs={"video": "VIDEO", "fps": "INT", "num_frames": "INT"},
        description="**流派A 第一阶段（Clean Plate）**：把 mask 区域从视频里"
                    "抹除，输出空无一人的纯净背景视频。method=telea（默认，"
                    "快）/ns（Navier-Stokes，大面积遮挡更自然）。dilate 保证"
                    "覆盖人物所有像素（原人残留会在下游生成时变成鬼影）。")
    def _op_bg_erase(video, mask, dilate=20, radius=15, method="telea"):
        import cv2
        import numpy as np
        from PIL import Image as _Image
        t0 = time.time()
        frames, fps = _video_to_frames(video, 0)
        masks, _ = _video_to_frames(mask, 0)
        n = len(frames)
        masks = (masks * ((n // len(masks)) + 1))[:n]
        flag = (cv2.INPAINT_NS if str(method).lower() == "ns"
                else cv2.INPAINT_TELEA)
        out = []
        for f, m in zip(frames, masks):
            img = np.asarray(f.convert("RGB"), dtype=np.uint8)
            mnp = np.asarray(m.convert("L").resize(f.size), dtype=np.uint8)
            mnp = (mnp > 127).astype(np.uint8) * 255
            kd = int(dilate or 0)
            if kd > 0:
                ker = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kd * 2 + 1, kd * 2 + 1))
                mnp = cv2.dilate(mnp, ker)
            res = cv2.inpaint(img, mnp, max(1, int(radius)), flag)
            out.append(_Image.fromarray(res))
        print(f"[video-bg-erase] done {n}帧 {time.time()-t0:.1f}s "
              f"dilate={dilate} radius={radius} method={method}", flush=True)
        return {"video": {"frames": out, "fps": fps},
                "fps": int(round(fps)), "num_frames": n}

    @registry.register(
        "video.pose_mask",
        inputs={"video": "VIDEO", "grow": "INT", "feather": "INT",
                "threshold": "INT"},
        outputs={"mask": "VIDEO", "fps": "INT", "num_frames": "INT"},
        description="**流派B 进攻掩码**：把 DWPose 骨架线（黑底白线）膨胀成"
                    "紧身轮廓软遮罩（grow px），用作条件图暗谷——只有这个紧身"
                    "形状被压暗，逼模型画瘦人物；原人宽轮廓区域保持亮窗帘由"
                    "模型补背景。feather>0 时转软边（L3 过渡）。")
    def _op_pose_mask(video, grow=40, feather=21, threshold=10):
        import cv2
        import numpy as np
        from PIL import Image as _Image
        t0 = time.time()
        frames, fps = _video_to_frames(video, 0)
        n = len(frames)
        out = []
        for f in frames:
            g = np.asarray(f.convert("L"), dtype=np.uint8)
            b = (g > int(threshold)).astype(np.uint8) * 255
            kg = int(grow or 0)
            if kg > 0:
                ker = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE, (kg * 2 + 1, kg * 2 + 1))
                b = cv2.dilate(b, ker)
            kf = int(feather or 0)
            if kf > 1:
                b = cv2.GaussianBlur(b, (kf | 1, kf | 1), 0)
            out.append(_Image.fromarray(b))
        print(f"[video-pose-mask] done {n}帧 {time.time()-t0:.1f}s "
              f"grow={grow} feather={feather}", flush=True)
        return {"mask": {"frames": out, "fps": fps},
                "fps": int(round(fps)), "num_frames": n}

    @registry.register(
        "video.depth",
        inputs={"video": "VIDEO", "invert": "BOOL"},
        outputs={"video": "VIDEO", "depth": "IMAGE", "fps": "INT"},
        description="视频深度估计（Depth-Anything-V2-Base）：VIDEO → 灰度深度"
                    "视频（近亮远暗）+ 首帧深度图。可单独预览深度、喂给"
                    "video-person-mask 的边界精修思路做验证，或给 ControlNet "
                    "depth 类节点当条件。invert=True 反转（远亮近暗）。")
    def _op_video_depth(video, invert=False):
        import numpy as np
        from PIL import Image as _Image

        t0 = time.time()
        frames, fps = _video_to_frames(video, 0)
        n = len(frames)
        depths = _estimate_depth_frames(frames, invert=bool(invert))
        vframes = [_Image.fromarray((d * 255).astype(np.uint8)).convert("RGB")
                   for d in depths]
        first = vframes[0].copy()
        print(f"[video-depth] done {n}帧 {time.time()-t0:.1f}s "
              f"invert={bool(invert)}", flush=True)
        return {"video": {"frames": vframes, "fps": fps},
                "depth": {"frames": [first], "fps": 1},
                "fps": int(round(fps))}

    @registry.register(
        "video.person_mask",
        inputs={"video": "VIDEO", "prompt": "STRING", "obj_pick": "STRING",
                "dilate": "INT", "max_frames": "INT",
                "shape": "STRING", "bbox_smooth": "INT",
                "mask_refine_radius": "INT", "mask_model": "STRING",
                "depth_refine": "INT", "mask_matting": "INT",
                "contour_smooth": "INT",
                "depth_video": "VIDEO",
                "motion_union": "BOOL"},
        outputs={"mask": "VIDEO", "fps": "INT", "num_frames": "INT",
                 "width": "INT", "height": "INT", "objects": "INT"},
        description="视频人物遮罩（mask_model 五选一）：sam3=文本直接分割"
                    "（1/8 网格）/ sam2=SAM3 首帧出框+SAM2 追踪（256 网格）/"
                    "depth=深度 Otsu 二值化（拓扑干净、无记忆毒素，要求人物"
                    "是最近前景）/ rvm=专用抠人模型（循环记忆，时序最稳，天然"
                    "软 alpha，免 prompt；抠全部前景人物）/ birefnet=单帧 SOTA"
                    "精修（边缘最锐，onnx 1024²，人物场景用 portrait 版）。"
                    "输出逐帧灰度遮罩"
                    "（255=选中；rvm 为连续软 alpha）。obj_pick: all=全部目标"
                    "合并 / best=仅最高分（仅 sam3/sam2 生效）")
    def _op_person_mask(video, prompt="person", obj_pick="all", dilate=12,
                        max_frames=0, shape="contour", bbox_smooth=5,
                        motion_union=False, mask_refine_radius=8,
                        mask_model="sam3", depth_refine=0,
                        mask_matting=0, depth_video=None,
                        contour_smooth=0):
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
        print(f"[video-mask] {n} 帧 {w}x{h} fps={fps:.2f} prompt={prompt!r} "
              f"model={mask_model} dilate={dilate} depth_refine={depth_refine} "
              f"matting={mask_matting} depth_video={'yes' if depth_video else 'no'}",
              flush=True)

        t0 = time.time()
        # **外部深度视频（2026-11-17）**：上游接 video-depth 时，深度只算
        # 一次且可单独预览，这里直接消费；否则各分支内部自估。
        _dframes = None
        if depth_video is not None:
            try:
                _dframes = _load_depth_video_frames(depth_video)
                print(f"[video-mask] 收到外部深度视频 {len(_dframes)} 帧",
                      flush=True)
            except Exception as _ed:  # noqa: BLE001
                print(f"[video-mask] 外部深度解析失败（{_ed}），改用内部估计",
                      flush=True)

        if str(mask_model).lower() == "sam2":
            # **SAM2 路径（2026-11-17）**：mask 精度 1/8 → 256 网格，边缘平滑，
            # 从根上缓解"mask 突出→替换后黑边"。后续仍走统一的
            # 时序/膨胀/精修流程（dilate、dilate、refine 与 SAM3 共用）。
            merged = _track_sam2(frames, prompt, obj_pick)
            merged = merged > 0.5
            _gf_r = int(mask_refine_radius or 0)
            import cv2 as _cv2s
            for _t in range(len(merged)):
                _m = merged[_t].astype(np.float32)
                if _gf_r > 0:
                    try:
                        _gray = np.asarray(frames[_t].convert("L"),
                                           dtype=np.float32)
                        _m = _guided_filter(_gray, _m, r=_gf_r, eps=1e-4)
                    except Exception:  # noqa: BLE001
                        pass
                _m = (_m > 0.5).astype("uint8")
                _m = _cv2s.morphologyEx(
                    _m, _cv2s.MORPH_CLOSE,
                    _cv2s.getStructuringElement(_cv2s.MORPH_ELLIPSE, (9, 9)))
                merged[_t] = _m > 0
            if motion_union in (True, "1", "true", "True", 1):
                merged = _motion_union(merged, frames)
            if int(depth_refine or 0) > 0:
                merged = _depth_refine(merged, frames, px=int(depth_refine),
                                       depth_maps=_dframes)
            merged = _dilate_masks(merged, int(dilate))
            if str(shape).lower() == "bbox":
                merged = _bbox_rectify(merged, smooth=int(bbox_smooth))
            _cs2 = int(contour_smooth or 0)
            if _cs2 > 0:
                merged = np.stack([_smooth_contour(merged[i], _cs2)
                                   for i in range(len(merged))])
                print(f"[video-mask] 轮廓平滑 win={_cs2}（SAM2 分支）",
                      flush=True)
            _mm = int(mask_matting or 0)
            if _mm > 0:
                _alpha = _trimap_matting(merged, frames, depth_maps=_dframes,
                                         px=_mm)
                mask_frames = [
                    Image.fromarray((np.clip(_alpha[i], 0, 1) * 255
                                     ).astype(np.uint8)) for i in range(n)]
                print(f"[video-mask] done(SAM2+trimap) "
                      f"{time.time()-t0:.1f}s 软 alpha 输出", flush=True)
                return {"mask": {"frames": mask_frames, "fps": fps},
                        "fps": int(round(fps)), "num_frames": n,
                        "width": w, "height": h, "objects": 1}
            mask_frames = [
                Image.fromarray(merged[i].astype(np.uint8) * 255)
                for i in range(n)]
            print(f"[video-mask] done(SAM2) {time.time()-t0:.1f}s "
                  f"shape={shape}", flush=True)
            return {"mask": {"frames": mask_frames, "fps": fps},
                    "fps": int(round(fps)), "num_frames": n,
                    "width": w, "height": h, "objects": 1}
        _mm_choice = str(mask_model).lower()
        if _mm_choice == "depth":
            # **深度二值化分支（2026-11-17）**：SAM 对 prompt/记忆敏感，遮挡
            # 与相似纹理时拓扑崩溃（粘连块/锯齿）。DA-V2 深度图本身层次分明
            # （人物 0.57 vs 背景 0.13），直接 Otsu 二值化 + 时序多数表决，
            # 拓扑绝对干净、无记忆毒素。适用前提：人物是最近的前景、
            # 背景纵深明显（近处杂物会被误包）。
            _dep = _dframes if _dframes is not None \
                else _estimate_depth_frames(frames)
            merged, thr = _depth_binary_masks(_dep)
            merged = _temporal_vote(merged, 3)
            print(f"[video-mask] 深度二值化 thr={thr:.3f}"
                  f"（全片统一 Otsu）+ 3帧多数表决", flush=True)
            merged = _dilate_masks(merged, int(dilate))
            _csd = int(contour_smooth or 0)
            if _csd > 0:
                merged = np.stack([_smooth_contour(merged[i], _csd)
                                   for i in range(len(merged))])
            mask_frames = [
                Image.fromarray(merged[i].astype(np.uint8) * 255)
                for i in range(n)]
            print(f"[video-mask] done(depth) {time.time()-t0:.1f}s "
                  f"平均覆盖率={float(merged.mean())*100:.1f}%", flush=True)
            return {"mask": {"frames": mask_frames, "fps": fps},
                    "fps": int(round(fps)), "num_frames": n,
                    "width": w, "height": h, "objects": 1}
        if _mm_choice == "rvm":
            # **RVM 抠像分支（2026-11-17）**：专用视频人物抠图（循环记忆），
            # 输出天然软 alpha——无需二值化/羽化/轮廓平滑（全部跳过），
            # 软边直接对接 wan.animate2 的软 alpha 合成与 latent 拉回权重。
            # dilate 语义保留（对连续 alpha 做 max 滤波=软膨胀）。
            _alpha, _wp = _matting_rvm(frames)
            print(f"[video-mask] RVM 抠像 {_alpha.shape} "
                  f"软alpha mean={float(_alpha.mean()):.3f} 权重={_wp}",
                  flush=True)
            return _emit_soft_mask(_alpha, dilate, fps, n, w, h,
                                   "rvm", t0)
        if _mm_choice == "birefnet":
            # **BiRefNet 分支（2026-11-17）**：单帧 SOTA 高精度抠图（双分辨率
            # 全局参考，边缘质量远超 SAM 系），1024² onnx 逐帧推理 + 中心
            # 加权时序平滑。与 RVM 分工：RVM=时序最稳（记忆）、BiRefNet=
            # 边缘最锐（单帧精修）；人物场景优先 portrait 版权重。
            _alpha, _wp = _matting_birefnet(frames)
            print(f"[video-mask] BiRefNet 抠图 {_alpha.shape} "
                  f"软alpha mean={float(_alpha.mean()):.3f} 权重={_wp}",
                  flush=True)
            return _emit_soft_mask(_alpha, dilate, fps, n, w, h,
                                   "birefnet", t0)
        bundle = _load_sam3()
        model, processor = bundle["model"], bundle["processor"]

        t0 = time.time()
        session = processor.init_video_session(
            video=frames, inference_device="cuda",
            dtype=torch.bfloat16)
        session = processor.add_text_prompt(session, prompt)

        # **float 置信度图（2026-11-16）**：全程保留 sigmoid 概率，最后才阈值化。
        # 一旦中途 `>0` 转 bool，1/8 网格的台阶就固化成 8px 方块（NEAREST 放大），
        # 后续任何滤波都只能圆化、消不掉。
        merged = np.zeros((n, h, w), dtype=np.float32)
        objects_seen = 0
        with torch.no_grad():
            for i in range(n):
                out = model(inference_session=session, frame_idx=i)
                # **必须传 original_sizes（2026-11-16 精度根因）**：
                # 不传时 postprocess 只能按 session 内部尺度输出，实测是
                # 原图的 1/8（768x1344 → 96x168），再用 NEAREST 放大回原尺寸
                # 就得到**8px 方块阶梯**边缘（用户反馈"蒙版没精确捕捉到人"）。
                # 传 [[h, w]] 后 mask 直接是逐像素精度。
                post = processor.postprocess_outputs(
                    session, out, original_sizes=[[h, w]])
                masks = post["masks"]       # (num_obj, H, W) bool
                scores = post["scores"]     # (num_obj,)
                if masks.shape[0] == 0:
                    continue
                objects_seen = max(objects_seen, int(masks.shape[0]))
                # **低分辨率 logits → 双线性上采样（2026-11-16 平滑根因）**：
                # `post["masks"]` 是模型内部 1/8 尺度经 **NEAREST** 放大到原尺寸
                # 的结果（8px 方块阶梯，任何后处理都消不干净）。这里改取原始
                # low-res logits 自己做 bilinear 插值 → 边界天然平滑，
                # 后续 guided filter 才真正有效。
                _low = None
                try:
                    _low = (out.get("obj_id_to_mask") if isinstance(out, dict)
                            else getattr(out, "obj_id_to_mask", None))
                except Exception:  # noqa: BLE001
                    _low = None
                if _low is not None and len(_low):
                    _ids = sorted(_low.keys())
                    _ups = []
                    for _oid in _ids:
                        _lg = _low[_oid]
                        if not torch.is_tensor(_lg):
                            continue
                        _lg = _lg.float()
                        if _lg.ndim == 2:
                            _lg = _lg[None]
                        # **先 area 下采样回低分辨率、再 bilinear 放大**
                        # （等效低通）：logits 送到这里时已经是模型内部 1/8
                        # 网格经 NEAREST 放大到原尺寸的"方块"，直接 bilinear
                        # 是空操作（尺寸已相同）。先降回 1/8 做区域平均把方块
                        # 抹平成真实置信度场，再 bilinear 升回原尺寸 →
                        # 边界是**连续渐变**，阈值化后即平滑轮廓。
                        _hh = max(1, h // 8)
                        _ww = max(1, w // 8)
                        _lg_s = torch.nn.functional.interpolate(
                            _lg[None], size=(_hh, _ww), mode="area")
                        _up = torch.nn.functional.interpolate(
                            _lg_s, size=(h, w), mode="bilinear",
                            align_corners=False)[0, 0]
                        _ups.append(_up)
                    if _ups:
                        _stack = torch.stack(_ups)      # (num_obj, h, w) logits
                        if str(obj_pick).lower() == "best" \
                                and _stack.shape[0] > 1:
                            _best = int(scores.argmax().item()) \
                                if scores is not None else 0
                            _stack = _stack[_best:_best + 1]
                        # logits → 概率（0~1），**不阈值化**，交给后面
                        # guided filter 做保边平滑后才二值化
                        _prob = torch.sigmoid(_stack).max(dim=0).values
                        fm_prob = _prob.cpu().numpy()
                        merged[i] = np.maximum(merged[i], fm_prob)
                        print(f"[video-mask] bilinear 概率图 "
                              f"{tuple(_ups[0].shape)} → {h}x{w} "
                              f"(obj={_stack.shape[0]}, float 保留)", flush=True)
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
                if not (_low is not None and len(_low)):
                    # 没拿到 logits 时的兜底：bool → 0/1 概率图
                    merged[i] = np.maximum(merged[i], fm.astype(np.float32))
                if i % 20 == 0 or i == n - 1:
                    cover = float((merged[i] > 0.5).mean()) * 100
                    print(f"[video-mask] frame {i+1}/{n} "
                          f"obj={masks.shape[0]} cover={cover:.1f}%",
                          flush=True)

        # **主轮廓平滑（2026-11-15）**：SAM3 锯齿边缘平滑（椭圆核闭运算），
        # 减少 mask 边界粗糙导致的生成暗过渡。
        import cv2 as _cv2m
        # **原图引导精修（2026-11-16）**：mask 原始有效分辨率是 1/8（SAM3 硬限），
        # 直接放大必是 8px 阶梯。这里用**原视频灰度做引导**把它吸附到真实轮廓：
        #   1) 先 float 化（相当于双线性，去掉 NEAREST 的方块感）
        #   2) guided filter（r 越大越平滑、越贴边缘）
        #   3) 阈值回二值 + 小核闭运算补孔
        _gf_r = int(mask_refine_radius or 0)
        for _t in range(len(merged)):
            _m = merged[_t].astype(np.float32)
            if _gf_r > 0:
                try:
                    _gray = np.asarray(frames[_t].convert("L"),
                                       dtype=np.float32)
                    _m = _guided_filter(_gray, _m, r=_gf_r, eps=1e-4)
                except Exception as _eg:  # noqa: BLE001
                    print(f"[video-mask] 引导精修跳过（{_eg}）", flush=True)
            _m = (_m > 0.5).astype("uint8")
            _m = _cv2m.morphologyEx(_m, _cv2m.MORPH_CLOSE,
                                    _cv2m.getStructuringElement(
                                        _cv2m.MORPH_ELLIPSE, (9, 9)))
            merged[_t] = _m > 0
        # **float → bool（2026-11-16）**：上面逐帧赋值虽写的是 bool，但数组
        # dtype 由 np.zeros(..., float32) 决定，赋值后仍是 float32 →
        # 下游 `~merged.any(...)` 会抛 "ufunc 'invert' not supported"。
        merged = merged > 0.5
        # **帧差运动兜底（2026-11-15 裙摆漏罩根因修复，在 dilate 前）**
        if motion_union in (True, "1", "true", "True", 1):
            merged = _motion_union(merged, frames)
        if int(depth_refine or 0) > 0:
            merged = _depth_refine(merged, frames, px=int(depth_refine),
                                   depth_maps=_dframes)
        merged = _dilate_masks(merged, int(dilate))
        _cs3 = int(contour_smooth or 0)
        if _cs3 > 0:
            merged = np.stack([_smooth_contour(merged[i], _cs3)
                               for i in range(len(merged))])
            print(f"[video-mask] 轮廓平滑 win={_cs3}（SAM3 分支）", flush=True)
        _mm3 = int(mask_matting or 0)
        if _mm3 > 0:
            _alpha3 = _trimap_matting(merged, frames, depth_maps=_dframes,
                                      px=_mm3)
            mask_frames = [Image.fromarray(
                (np.clip(_alpha3[i], 0, 1) * 255).astype(np.uint8))
                for i in range(n)]
            print(f"[video-mask] done(SAM3+trimap) "
                  f"{time.time()-t0:.1f}s 软 alpha 输出", flush=True)
            return {"mask": {"frames": mask_frames, "fps": fps},
                    "fps": int(round(fps)), "num_frames": n,
                    "width": w, "height": h, "objects": objects_seen}
        # **矩形化（2026-11-15 官方配方对齐）**：Wan2.2 replace 官方 mask
        # 是人物 bbox 矩形（get_aug_mask w1h1），不是精细轮廓——消灭轮廓
        # 黑影与"人物图层感"的关键。shape=bbox 时启用。
        if str(shape).lower() == "bbox":
            merged = _bbox_rectify(merged, smooth=int(bbox_smooth))
        empty = int((~merged.any(axis=(1, 2))).sum())
        if empty:
            print(f"[video-mask] 警告：{empty}/{n} 帧无遮罩（目标未检出/跟丢）",
                  flush=True)
        mask_frames = [Image.fromarray(merged[i].astype(np.uint8) * 255)
                       for i in range(n)]
        print(f"[video-mask] done {time.time()-t0:.1f}s shape={shape} "
              f"objects_peak={objects_seen}", flush=True)
        return {"mask": {"frames": mask_frames, "fps": fps},
                "fps": int(round(fps)), "num_frames": n,
                "width": w, "height": h, "objects": objects_seen}
