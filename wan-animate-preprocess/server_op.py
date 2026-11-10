"""video.* 算子族：视频 I/O（load / to_frames / encode）+ 视频到视频编辑
（repaint 逐帧重绘、sample_v2v 原生 v2v 采样）。

成员节点：load-video / video-to-frames / frames-to-video / video-repaint /
video-v2v。同族同文件：插件上传是单文件机制，同族多算子同文件可避免跨文件重名
409（同 flux / seg 家族模式）；集中放 _common 由 sync-common.py 分发，防多副本漂移。

VIDEO 对象契约（app/video.py）：put 时即编码 mp4 落盘，仓库内为
{"path", "fps", "num_frames"}（预编码直通见 object_store._adopt_video_file）。
**因此所有消费 VIDEO 的算子都只能从 path 解码帧序列**——帧本体不进控制通道。

解码/编码统一走 imageio（imageio-ffmpeg 自带静态 ffmpeg 二进制，容器内无需
apt 安装 ffmpeg）；imageio 延迟导入：未安装时仅视频算子不可用，其余不受影响。

分钟级任务（repaint / sample_v2v），调用方应经异步 job（POST /jobs）执行；
进度与预览经 app.execution + app.preview 接线：
- repaint 逐帧解码出的是**真图**，直接 push_pil 做实时预览（画布能看到换人效果）
- sample_v2v 中间结果是 3D latent（无法廉价投影），沿用 video.sample 的进度卡片
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "ee8e48f9cd95"
OP_FAMILY = "video_flow"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

import inspect
import os
import random
import time

from PIL import Image

from app import execution, ops, preview


# ---------- 解码/归一化（纯 PIL + imageio，无 GPU 依赖） ----------

def _open_reader(path):
    """打开视频文件（imageio reader）；路径不存在/格式不支持时给可操作报错。

    先查存在性再 import imageio：路径写错是最常见的一类失误，不该被
    「没装 imageio」的 ImportError 盖掉（也便于无 imageio 环境单测）。
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"video file not found: {path}")
    import imageio.v2 as imageio  # 延迟导入
    try:
        return imageio.get_reader(path)
    except Exception as e:  # noqa: BLE001 - 编解码器报错形态多样
        raise ValueError(f"cannot open video '{path}': {e}")


def _probe(path):
    """只读元数据：返回 (fps, num_frames, width, height)；帧数取不到时按 0。"""
    reader = _open_reader(path)
    try:
        meta = reader.get_meta_data() or {}
        fps = float(meta.get("fps") or 24.0)
        size = meta.get("size") or (0, 0)
        n = 0
        try:
            n = int(reader.count_frames())
        except Exception:  # noqa: BLE001 - 部分格式不支持帧数查询
            duration = meta.get("duration")
            if duration:
                n = int(round(float(duration) * fps))
    finally:
        reader.close()
    return fps, max(0, n), int(size[0] or 0), int(size[1] or 0)


def frame_step(src_fps, fps=0.0, stride=1):
    """抽帧步长与有效 fps（纯函数，便于无解码器单测）。

    步长 = max(stride, round(源fps/目标fps))：目标 fps 只有低于源 fps 才有意义
    （不插帧——没有中间帧可插），有效 fps = 源 fps / 步长。
    """
    step = max(1, int(stride or 1))
    if fps and fps > 0 and src_fps > float(fps):
        step = max(step, int(round(src_fps / float(fps))))
    return step, (src_fps / step if src_fps > 0 else 24.0)


def read_video(path, start=0, end=0, stride=1, max_frames=0, fps=0.0):
    """mp4 → (PIL 帧列表, 有效 fps, 源 fps)。

    抽帧口径（与 ComfyUI Load Video / VHS 一致）：
    - start/end：帧序号区间 [start, end)，end<=0 表示到尾
    - fps>0 且小于源 fps 时**降采样**（步长 = round(源fps/目标fps)），有效 fps
      随之变为 源fps/步长（不做插帧——没有中间帧可插）
    - stride：额外抽帧步长（与 fps 降采样取较大步长）
    - max_frames：截断到前 N 帧
    帧统一转 RGB（H.264 是 RGB；个别解码器给 RGBA/灰度，统一口径避免下游 vae
    encode 拿到 4 通道）。
    """
    reader = _open_reader(path)
    try:
        meta = reader.get_meta_data() or {}
        src_fps = float(meta.get("fps") or 24.0)
        step, out_fps = frame_step(src_fps, fps, stride)
        frames = []
        for i, arr in enumerate(reader):
            if end and end > 0 and i >= end:
                break
            if i < int(start or 0) or i % step:
                continue
            frames.append(Image.fromarray(arr).convert("RGB"))
            if max_frames and len(frames) >= int(max_frames):
                break
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"video '{path}' 解码出 0 帧（检查 start={start} / end={end} "
                         f"/ stride={stride} 与源帧数是否匹配）")
    return frames, out_fps, src_fps


def fit_frames(frames, width=0, height=0):
    """等比/定尺缩放帧序列：两者都给 → 目标尺寸；只给一个 → 按该边等比。

    输出尺寸取整到**偶数**：H.264 要求偶数边长（encode_mp4 用 macro_block_size=None
    保持原始尺寸，imageio 会给奇数边补 1px），主动取偶让落盘尺寸与预期一致。
    """
    if not width and not height:
        return frames
    out = []
    for img in frames:
        w, h = img.size
        if width and height:
            tw, th = int(width), int(height)
        elif width:
            tw = int(width)
            th = max(2, int(round(h * tw / w)))
        else:
            th = int(height)
            tw = max(2, int(round(w * th / h)))
        tw, th = max(2, tw - tw % 2), max(2, th - th % 2)
        out.append(img if (tw, th) == (w, h) else img.resize((tw, th), Image.LANCZOS))
    return out


def video_frames_of(video):
    """VIDEO 对象 → (帧列表, fps)。仓库内是 {"path",...}（落盘形态），
    但也兼容无 video_dir 的内存形态 {"frames":[...]}（单测/无盘部署）。"""
    if isinstance(video, dict) and video.get("frames"):
        frames = list(video["frames"])
        return frames, float(video.get("fps") or 24.0)
    if isinstance(video, dict) and video.get("path"):
        frames, out_fps, _ = read_video(video["path"])
        return frames, float(video.get("fps") or out_fps)
    raise ValueError(f"VIDEO 对象形态非法：{type(video).__name__}"
                     f"{'（缺 path/frames）' if isinstance(video, dict) else ''}")


def _subsample(video, frames, fps, start=0, end=0, stride=1, max_frames=0):
    """带抽帧参数时重解码（有 mp4 就按原片抽，保证帧号口径与源一致）；内存形态
    VIDEO（无 path，仅单测/无盘部署）退化为对已解码帧做等价切片。"""
    path = video.get("path") if isinstance(video, dict) else None
    if path:
        frames, out_fps, _ = read_video(path, start, end, stride, max_frames)
        return frames, out_fps
    picked = frames[int(start or 0):(int(end) if end else None)]
    step = max(1, int(stride or 1))
    if step > 1:
        picked = picked[::step]
    if max_frames:
        picked = picked[:int(max_frames)]
    if not picked:
        raise ValueError("抽帧结果为空（检查 start/end/stride/max_frames）")
    return picked, fps / step


def as_frame_list(images):
    """IMAGE 对象 → PIL 帧列表（单张 / 批量 list 都接受）。"""
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, (list, tuple)) and images \
            and all(isinstance(i, Image.Image) for i in images):
        return list(images)
    raise ValueError(f"IMAGE 对象形态非法：期望 PIL 图或 PIL 图列表，"
                     f"got {type(images).__name__}")


# ---------- 逐帧回调（job 进度 + 预览） ----------

def _frame_cb(preview_every, total, decode=None):
    """逐帧回调：上报 job 进度；preview_every>0 时录预览帧。

    decode 给了就录**解码后的真图**（repaint 用：画布直接看到换人效果），
    否则录纯渲染进度卡片（sample_v2v 的 3D latent 无法廉价投影）。
    """
    state = {"every": max(1, int(preview_every or 1))}

    def on_frame(idx, frame=None):
        job = execution.current()
        if job is not None:
            job.set_progress(idx, total)
        if not preview_every:
            return
        if job is None:
            return  # 同步 /op 调用无 job 上下文：不录预览
        if idx % state["every"] and idx != total:
            return
        rec = preview.recorder_for(job.id, 1)
        if frame is not None:
            rec.push_pil(frame, idx / max(1, total))
        else:
            rec.push_pil(preview.progress_card(idx, total), idx / max(1, total))
    return on_frame


def effective_steps(steps, strength, enforce=False):
    """有效去噪步数（纯函数，便于无 GPU 单测）。

    为什么需要它（真机事故 2026-10-14）：`AnimateDiffVideoToVideoPipeline` 的
    `strength` **不是"重绘强度"而是"时间轴截断比例"**——
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = timesteps[t_start:]        # 只保留尾部
    ⇒ `steps=3, strength=0.6` 实际只跑 **1 步**（DDIM linspace 3 步 = [999,666,333]，
    截断后只剩 t=333）；实测 ρ(源,出)=0.80~0.86，提示词完全没生效（"没有油画风"）。
    3 步时任意 0.34≤strength<1 都塌成 1 步；strength<1/3 直接 0 步（原样返回）。

    `enforce=True`（管道同名的 `enforce_inference_steps` 开关）走另一分支：
        denoising_inference_steps = int(num_inference_steps / strength)
        timesteps = retrieve_timesteps(..., denoising_inference_steps)[-num_inference_steps:]
    ⇒ steps 是**实际执行步数**、strength 只决定从 sigma 排布哪一段开始（ComfyUI 语义，
    对齐 `ksample` 的 denoise），此时有效步数恒等于 steps。
    """
    n = max(1, int(steps))
    if enforce:
        return n
    return min(max(0, int(n * float(strength))), n)


def _report_effective_steps(tag, steps, strength, enforce=False, min_ok=2):
    """把"有效步数"打进日志，并对塌缩给出可操作告警（不静默）。

    0 步：管道会原样返回输入（等于没做事）⇒ 调用方应报错；
    1 步：几乎不改变内容（提示词基本无效）⇒ 告警 + 给出配比建议。
    """
    eff = effective_steps(steps, strength, enforce)
    hint = (f"steps×strength={int(steps) * float(strength):.1f}（需 ≥{min_ok}；"
            f"建议 steps=25+strength=0.7 ⇒ 17 步），或设 enforce_inference_steps=true "
            f"用 ComfyUI 语义（steps 即实际步数）")
    print(f"[{tag}] steps={int(steps)} strength={strength} enforce={bool(enforce)} "
          f"⇒ 有效去噪步数={eff}", flush=True)
    if eff < 1:
        return eff, f"有效去噪步数=0（strength<1/steps）：管道会原样返回输入、什么都不会变。{hint}"
    if eff == 1:
        return eff, f"有效去噪步数=1：提示词/风格基本不会生效。{hint}"
    return eff, None


def align_vae_dtype(pipe):
    """把 VAE 权重与其"解码包装器"的 dtype 约定对齐（幂等，防 dtype 混用崩溃）。

    服务端组合管在 Pascal 上给 `vae.decode` 包一层 `_decode_fp32`（VAE_FP32=1，
    防黑图），约定是「VAE 权重 fp32 + 解码入口统一转 fp32」。实测存在包装器在、
    权重却停在 fp16 的状态：包装器把 latent 转成 fp32 喂给 fp16 权重 ⇒ conv2d 报
    `Input type (float) and bias type (c10::Half)`（t2v/v2v 都在 decode 阶段崩）。
    这里按包装器语义把权重对齐到 fp32；没有该包装器时不动（不擅自改用户精度）。
    """
    import torch  # 延迟导入

    vae = getattr(pipe, "vae", None)
    if vae is None:
        return
    # 判定"解码入口被包装过"：原生 decode 是模块绑定方法；包装器是普通函数。
    # 按名称判定太脆（调试层/多层包装都会改名），按 ismethod 判定与实现无关。
    if inspect.ismethod(vae.decode):
        return  # 没被包装：dtype 与解码路径自洽，不动（不擅自改用户精度）
    if vae.dtype != torch.float32:
        print(f"[video] VAE 权重 {vae.dtype} 与 fp32 解码包装器不一致 → 对齐到 fp32",
              flush=True)
        vae.to(torch.float32)


class _VAEDtypeShim:
    """VAE 精度对齐外壳：对外声明计算精度，内部按 VAE 真实精度编解码。

    为什么需要（真机定位 2026-10-13，GTX 1080 + diffusers 0.35.2）：
    视频 v2v 管道在**内部**自己 VAE 编解码（`prepare_latents` → `encode_video`、
    `decode_latents`），并且会按管道计算精度 `self.dtype`（取 unet dtype）动手：
    `prepare_latents` 里 `if self.vae.config.force_upcast: ... self.vae.to(dtype)`
    会把 VAE **转成 fp16**；而服务端 Pascal 组合管把 VAE 权重留在 fp32 并用
    `_decode_fp32` 包装解码入口（VAE_FP32 防黑图）⇒ 包装器把 latent 转 fp32 喂给
    fp16 权重，conv2d 报 `Input type (float) and bias type (c10::Half)`，
    t2v/v2v 都在 decode 阶段崩。

    外壳把两件事隔开：① `config.force_upcast` 对外恒为 False（管道不再擅自改 VAE
    精度，真实 config 不动）；② `dtype` 对外是计算精度、encode/decode 内部按 VAE
    真实精度算并转换入出参。VAE 本体与其它管道共享，权重/精度不复制不改写。
    """

    def __init__(self, vae, compute_dtype):
        object.__setattr__(self, "_vae", vae)
        object.__setattr__(self, "_compute_dtype", compute_dtype)
        object.__setattr__(self, "config", _ConfigShim(vae.config))

    @property
    def dtype(self):
        return self._compute_dtype

    @property
    def device(self):
        return getattr(self._vae, "device", None)

    def __getattr__(self, name):  # 其余属性/子模块直通（image_processor 等）
        return getattr(self._vae, name)

    def to(self, *args, **kwargs):
        """只透传设备搬移，**忽略 dtype**：外壳存在的意义就是不让管道改 VAE 精度。"""
        import torch  # 延迟导入
        kwargs.pop("dtype", None)
        rest = tuple(a for a in args if not isinstance(a, torch.dtype))
        if rest or kwargs:
            self._vae.to(*rest, **kwargs)
        return self

    def encode(self, pixel, *args, **kwargs):
        out = self._vae.encode(pixel.to(self._vae.dtype), *args, **kwargs)
        return _CastOutput(out, self._compute_dtype)

    def decode(self, z, *args, **kwargs):
        return self._vae.decode(z.to(self._vae.dtype), *args, **kwargs)


class _ConfigShim:
    """config 外壳：force_upcast 对外恒 False，其余直通（scaling_factor 等）。"""

    def __init__(self, cfg):
        object.__setattr__(self, "_cfg", cfg)

    @property
    def force_upcast(self):
        return False

    def __getattr__(self, name):
        return getattr(self._cfg, name)


class _CastOutput:
    """encode 结果外壳：latent_dist 的 sample()/mode() 结果转回计算精度。"""

    def __init__(self, out, dtype):
        object.__setattr__(self, "_out", out)
        object.__setattr__(self, "_dtype", dtype)

    def __getattr__(self, name):
        return getattr(self._out, name)

    @property
    def latent_dist(self):
        return _CastOutput(self._out.latent_dist, self._dtype)

    def sample(self, *args, **kwargs):
        return self._out.sample(*args, **kwargs).to(self._dtype)

    def mode(self, *args, **kwargs):
        return self._out.mode(*args, **kwargs).to(self._dtype)


def as_v2v_pipe(pipe):
    """把 AnimateDiff **t2v** 组合管改视成 v2v 管道（共享同一批组件对象，不复制权重）。

    为什么在算子层做：motion-loader 组合出的是 AnimateDiffPipeline（__call__ 无 video
    参数），而整段 v2v 需要 AnimateDiffVideoToVideoPipeline。两者的组件完全一致
    （vae/text_encoder/tokenizer/unet/motion_adapter/scheduler/feature_extractor/
    image_encoder），只是 __call__ 不同 ⇒ 按组件构造一个 v2v 视图即可，**无需重新加载
    模型、不额外占显存、也不用改服务端核心加载层**（既有 motion-loader 组合管直接获得
    v2v 能力）。

    已经是接受 video 参数的管道（原生 v2v，如 Wan/LTX 系 v2v）时原样返回。
    """
    if "video" in inspect.signature(pipe.__call__).parameters:
        return pipe
    if not hasattr(pipe, "motion_adapter"):
        raise ValueError(
            f"{type(pipe).__name__} 不支持视频到视频：__call__ 无 video 参数，"
            f"也不是 AnimateDiff 运动组合管（motion-loader 产物）。"
            f"请改用 AnimateDiff 组合管 + 本算子，或改用 video.repaint 逐帧重绘")
    try:
        from diffusers import AnimateDiffVideoToVideoPipeline as V2VPipe
    except Exception as e:  # noqa: BLE001 - 版本过旧/缺类
        raise ValueError(
            f"当前 diffusers 不含 AnimateDiffVideoToVideoPipeline（{e}）："
            f"请升级 diffusers（>=0.24）后重试")
    comps = {k: getattr(pipe, k) for k in
             ("vae", "text_encoder", "tokenizer", "unet", "motion_adapter",
              "scheduler", "feature_extractor", "image_encoder")
             if hasattr(pipe, k)}
    # VAE 与计算精度不一致（或 config 声明 force_upcast）时套精度对齐外壳：
    # 管道会按 self.dtype 把 VAE 转成 fp16，破坏 fp32 防黑图 + 解码包装器约定
    vae, unet = comps.get("vae"), comps.get("unet")
    if vae is not None and unet is not None:
        need = (vae.dtype != unet.dtype
                or bool(getattr(getattr(vae, "config", None), "force_upcast", False)))
        if need:
            comps["vae"] = _VAEDtypeShim(vae, unet.dtype)
            print(f"[video.sample_v2v] VAE 精度对齐外壳：vae={vae.dtype} "
                  f"compute={unet.dtype} force_upcast="
                  f"{bool(getattr(getattr(vae, 'config', None), 'force_upcast', False))}",
                  flush=True)
    try:
        v2v = V2VPipe(**comps)
    except Exception as e:  # noqa: BLE001 - 组件契约不符
        raise ValueError(f"v2v 管道视图构造失败（组件不兼容）：{e}")
    try:
        v2v.set_progress_bar_config(disable=True)
    except Exception:  # noqa: BLE001 - 仅影响进度条输出
        pass
    print(f"[video.sample_v2v] AnimateDiff t2v 组合管 -> v2v 视图（共享组件，"
          f"不额外占显存）", flush=True)
    return v2v


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    # ---------- 视频 I/O ----------

    @registry.register(
        "video.load",
        inputs={"path": "STRING", "start": "INT", "end": "INT", "stride": "INT",
                "max_frames": "INT", "fps": "FLOAT", "width": "INT",
                "height": "INT"},
        outputs={"video": "VIDEO", "fps": "INT", "num_frames": "INT",
                 "width": "INT", "height": "INT"},
        description="Load Video：把服务端可读的视频文件（POST /videos 上传端点落盘的"
                    "路径，或服务端本地路径）登记为 VIDEO 对象，输出 fps/帧数/宽高。"
                    "不给 start/end/stride/max_frames/fps/width/height 时**原字节直通**"
                    "（不重编码，无损）；给了才解码归一化后重编码（降采样/截断/缩放）。")
    def _op_video_load(path, start=0, end=0, stride=1, max_frames=0, fps=0.0,
                       width=0, height=0):
        path = os.path.expanduser(str(path or ""))
        if not path:
            raise ValueError("video.load: path 不能为空")
        normalized = bool(start or end or (stride and int(stride) > 1)
                          or max_frames or (fps and float(fps) > 0)
                          or width or height)
        if not normalized:
            src_fps, n, w, h = _probe(path)
            out_fps = int(round(src_fps))
            print(f"[video.load] passthrough {path} fps={out_fps} "
                  f"frames={n} {w}x{h}", flush=True)
            return {"video": {"path": path, "fps": out_fps, "num_frames": n},
                    "fps": out_fps, "num_frames": n, "width": w, "height": h}
        frames, out_fps, src_fps = read_video(path, start, end, stride,
                                              max_frames, fps)
        frames = fit_frames(frames, width, height)
        out_fps_i = max(1, int(round(out_fps)))
        print(f"[video.load] normalized {path} src_fps={src_fps:.3f} -> "
              f"fps={out_fps_i} frames={len(frames)} "
              f"{frames[0].width}x{frames[0].height}", flush=True)
        return {"video": {"frames": frames, "fps": out_fps_i},
                "fps": out_fps_i, "num_frames": len(frames),
                "width": frames[0].width, "height": frames[0].height}

    @registry.register(
        "video.to_frames",
        inputs={"video": "VIDEO", "start": "INT", "end": "INT", "stride": "INT",
                "max_frames": "INT"},
        description="Video to Frames：VIDEO → 帧序列（IMAGE 批量对象，按 index 取单帧；"
                    "GET /images/{id}?index=n），并回传 fps/帧数。抽帧口径同 video.load"
                    "（start/end/stride/max_frames）。",
        outputs={"frames": "IMAGE", "fps": "INT", "count": "INT"})
    def _op_video_to_frames(video, start=0, end=0, stride=1, max_frames=0):
        frames, fps = video_frames_of(video)
        if start or end or (stride and int(stride) > 1) or max_frames:
            frames, fps2, _ = read_video(video["path"], start, end, stride,
                                         max_frames)
            fps = fps2
        print(f"[video.to_frames] {len(frames)} frames fps={int(round(fps))}",
              flush=True)
        return {"frames": frames, "fps": max(1, int(round(fps))),
                "count": len(frames)}

    @registry.register(
        "video.encode",
        inputs={"frames": "IMAGE", "fps": "INT"},
        description="Video Encode：帧序列（IMAGE 批量对象，或单张）→ VIDEO（put 时编码 "
                    "mp4）。帧来自 video.to_frames 或任何逐帧处理链的产物；"
                    "帧尺寸需一致（H.264 单流要求同分辨率）。",
        outputs={"video": "VIDEO", "num_frames": "INT"})
    def _op_video_encode(frames, fps=24):
        imgs = as_frame_list(frames)
        size = imgs[0].size
        for i, im in enumerate(imgs):
            if im.size != size:
                raise ValueError(
                    f"video.encode: 帧尺寸不一致（第 {i} 帧 {im.size} != 首帧 {size}）；"
                    f"请先把所有帧缩放到同一尺寸")
        fps = max(1, int(fps or 24))
        print(f"[video.encode] {len(imgs)} frames fps={fps} {size[0]}x{size[1]}",
              flush=True)
        return {"video": {"frames": imgs, "fps": fps}, "num_frames": len(imgs)}

    # ---------- 视频到视频 ----------

    @registry.register(
        "video.repaint",
        inputs={"model": "MODEL", "pos": "COND", "neg": "COND", "video": "VIDEO",
                "strength": "FLOAT", "steps": "INT", "cfg": "FLOAT", "seed": "INT",
                "sampler_name": "STRING", "scheduler": "STRING",
                "start_at_step": "INT", "end_at_step": "INT",
                "start": "INT", "end": "INT", "stride": "INT", "max_frames": "INT",
                "control_net": "CONTROL_NET", "control_frames": "IMAGE",
                "control_strength": "FLOAT", "preview_every": "INT"},
        outputs={"video": "VIDEO", "seed": "INT", "num_frames": "INT"},
        description="Video Repaint（逐帧重绘，v2v 主力）：每帧 VAE 编码 → 以 strength "
                    "为 denoise 重采样 → 解码，帧间保持原运动（无运动模块时的通用 v2v）。"
                    "**人物/风格替换**：model 接 ipadapter.apply 的 ModelRef（带 .ipa："
                    "采样每步注入参考图 embeds ⇒ 逐帧都带同一身份/风格）；pos/neg 为文本"
                    "条件。control_net（可选）按帧注入结构条件，不给 control_frames 时用"
                    "源帧自身（适合 tile/lineart 类 CN，canny/depth 类需自备条件图）。"
                    "Seed 只随机一次后逐帧继承（抑制帧间闪烁）。分钟级任务，请经 POST /jobs "
                    "执行；preview_every>0 时逐帧收录解码结果供画布实时预览，"
                    "/interrupt 可取消")
    def _op_video_repaint(model, pos, neg, video, strength=0.6, steps=20, cfg=7.0,
                          seed=-1, sampler_name="", scheduler="",
                          start_at_step=0, end_at_step=0, start=0, end=0,
                          stride=1, max_frames=0, control_net=None,
                          control_frames=None, control_strength=1.0,
                          preview_every=0):
        pipe, _patches = ops.resolve_pipe(model)
        frames, fps = video_frames_of(video)
        if start or end or (stride and int(stride) > 1) or max_frames:
            frames, fps = _subsample(video, frames, fps, start, end, stride,
                                     max_frames)
        hints = as_frame_list(control_frames) if control_frames is not None else None
        if hints is not None and control_net is None:
            raise ValueError("video.repaint: 给了 control_frames 就必须同时给 "
                             "control_net（否则条件图无处生效）")
        total = len(frames)
        # 逐帧重绘的有效步数由核心 resolve_span 决定（denoise=strength）：
        # start = min(round(steps×(1-strength)), steps-1) ⇒ 实际执行 steps-start 步。
        # 与 v2v 同理，配比过小等于"几乎不重绘"，这里显式报出来（不静默）。
        try:
            _st, _en = ops.resolve_span(int(steps), float(strength))
        except ValueError as e:  # denoise 太小 ⇒ 空区间
            raise ValueError(f"video.repaint: strength={strength} 与 steps={steps} "
                             f"推出的步区间为空（{e}）：请提高 strength 或 steps") from e
        print(f"[video.repaint] {total} frames strength={strength} steps={steps} "
              f"⇒ 有效去噪步数={_en - _st} cfg={cfg} "
              f"cn={'yes' if control_net is not None else 'no'}", flush=True)
        if _en - _st < 2:
            print(f"[video.repaint] WARN: 有效去噪步数={_en - _st}（<2）：重绘幅度很小，"
                  f"建议 steps×strength≥12（例 steps=24 + strength=0.6）", flush=True)
        on_frame = _frame_cb(preview_every, total)
        # 采样旋钮：空串走 ops.sample 默认（保持与 ksampler 同口径）
        kw = {"steps": int(steps), "cfg": float(cfg),
              "denoise": float(strength), "start_at_step": int(start_at_step),
              "end_at_step": int(end_at_step),
              "interrupt_check": execution.check_cancelled}
        if sampler_name:
            kw["sampler_name"] = sampler_name
        if scheduler:
            kw["scheduler"] = scheduler
        out_frames = []
        for i, frame in enumerate(frames):
            execution.check_cancelled()
            base = ops.vae_encode(pipe, frame)["latent"]
            control = None
            if control_net is not None:
                hint = hints[i % len(hints)] if hints else frame
                control = ops.controlnet_apply(control_net, hint,
                                               strength=float(control_strength))
            out = ops.sample(model, pos, neg, base, seed=seed, control=control, **kw)
            seed = out["seed"]  # 首帧定种后逐帧继承（-1 只随机一次）
            img = ops.vae_decode(pipe, out["latent"])["image"]
            img = img[0] if isinstance(img, list) else img
            out_frames.append(img)
            on_frame(i + 1, frame=img)
            print(f"[video.repaint] frame {i + 1}/{total} seed={seed}", flush=True)
        return {"video": {"frames": out_frames, "fps": max(1, int(round(fps)))},
                "seed": int(seed), "num_frames": len(out_frames)}

    @registry.register(
        "video.sample_v2v",
        inputs={"model": "MODEL", "video": "VIDEO", "prompt": "STRING",
                "neg_prompt": "STRING", "strength": "FLOAT", "steps": "INT",
                "cfg": "FLOAT", "seed": "INT", "width": "INT", "height": "INT",
                "num_frames": "INT", "fps": "INT", "decode_chunk_size": "INT",
                "preview_every": "INT", "enforce_inference_steps": "BOOL"},
        outputs={"video": "VIDEO", "seed": "INT"},
        description="Video Sample（v2v）：整段视频进运动/视频模型重采样（时间一致性好"
                    "于逐帧重绘）。model 可直接接 motion-loader 的 t2v 组合管——算子按"
                    "组件构造 AnimateDiffVideoToVideoPipeline 视图（共享权重，不额外"
                    "占显存、不重载模型）；原生 v2v 管道（Wan/LTX 系）原样使用。"
                    "非 v2v 且非 AnimateDiff 组合管的输入会**明确报错**，不会退化成 "
                    "t2v。⚠️ strength 是**时间轴截断比例**（有效步数 = int(steps×strength)），"
                    "steps=3/strength=0.6 只跑 1 步 ⇒ 提示词无效（真机事故）；"
                    "要出效果需 steps×strength≥12（例 25×0.7=17 步），或把 "
                    "enforce_inference_steps=true 打开（ComfyUI 语义：steps 即实际步数、"
                    "strength 只缩放 sigma 排布）。有效步数=0 时报错、=1 时告警，不静默。"
                    "分钟级任务，请经 POST /jobs 执行；preview_every>0 时留进度卡片帧；"
                    "/interrupt 可取消")
    def _op_video_sample_v2v(model, video, prompt="", neg_prompt="", strength=0.6,
                             steps=25, cfg=7.5, seed=-1, width=0, height=0,
                             num_frames=0, fps=0, decode_chunk_size=0,
                             preview_every=0, enforce_inference_steps=False):
        import torch  # 延迟导入：仅本算子需要

        base_pipe, patches = ops.resolve_pipe(model)
        if patches:
            print("[video.sample_v2v] lora patches: "
                  + ", ".join(f"{n}@{w}" for n, w in patches), flush=True)
        ops._enable_adapters(base_pipe, patches)
        # t2v 组合管 → v2v 视图（组件共享）；原生 v2v 管道原样用
        pipe = as_v2v_pipe(base_pipe)
        align_vae_dtype(pipe)  # 防「fp16 权重 + fp32 解码包装器」dtype 混用崩溃
        sig = inspect.signature(pipe.__call__).parameters
        cls_name = type(pipe).__name__
        eff, warn = _report_effective_steps("video.sample_v2v", steps, strength,
                                            enforce_inference_steps)
        if eff < 1:
            raise ValueError(warn)  # 0 步 = 原样返回输入，属误用而非"重绘强度小"
        if warn:
            print(f"[video.sample_v2v] WARN: {warn}", flush=True)

        frames, src_fps = video_frames_of(video)
        if num_frames and int(num_frames) > 0:
            frames = frames[:int(num_frames)]
        out_fps = int(fps) if fps and int(fps) > 0 else max(1, int(round(src_fps)))
        # 尺寸：不给就跟随源片（否则 v2v 管道会退回 unet 默认 512² 把片子放大，
        # 与"0=用源视频尺寸"的节点口径不符）。取 8 的整数倍（VAE 下采样倍数），
        # 避免 latent 取整后与源尺寸不一致导致解码帧尺寸漂移。
        if not width or not height:
            width = int(width or frames[0].width)
            height = int(height or frames[0].height)
        width = max(64, int(width) - int(width) % 8)
        height = max(64, int(height) - int(height) % 8)
        seed = seed if seed >= 0 else random.randint(0, 2**32 - 1)
        gen = torch.Generator(device=ops.exec_device_of(pipe)).manual_seed(seed)
        total = len(frames)
        print(f"[video.sample_v2v] {cls_name} frames={total} {width}x{height} "
              f"strength={strength} steps={steps} cfg={cfg} seed={seed}", flush=True)
        on_frame = _frame_cb(preview_every, int(steps))

        def on_step(p, i, t, cb_kwargs):
            execution.check_cancelled()
            on_frame(i + 1)
            return cb_kwargs

        candidates = {
            "video": frames,
            "prompt": prompt or None,
            "negative_prompt": neg_prompt or None,
            "strength": float(strength),
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg),
            "generator": gen,
            "output_type": "pil",
        }
        candidates["width"] = int(width)
        candidates["height"] = int(height)
        if decode_chunk_size and int(decode_chunk_size) > 0:
            candidates["decode_chunk_size"] = int(decode_chunk_size)
        candidates["enforce_inference_steps"] = bool(enforce_inference_steps)
        kwargs = {k: v for k, v in candidates.items() if k in sig and v is not None}
        if enforce_inference_steps and "enforce_inference_steps" not in sig:
            print(f"[video.sample_v2v] {cls_name} 无 enforce_inference_steps 参数："
                  f"按默认（截断语义）执行", flush=True)
        dropped = sorted(set(candidates) - set(kwargs)
                         - {"prompt", "negative_prompt", "video"})
        if dropped:
            print(f"[video.sample_v2v] {cls_name} 不支持的参数已忽略: {dropped}",
                  flush=True)
        t0 = time.time()
        try:
            call = dict(kwargs)
            if "callback_on_step_end" in sig:
                call["callback_on_step_end"] = on_step
                if "callback_on_step_end_tensor_inputs" in sig:
                    call["callback_on_step_end_tensor_inputs"] = ["latents"]
            out = pipe(**call)
        finally:
            if patches:
                try:
                    base_pipe.disable_lora()
                except Exception:  # noqa: BLE001 - 无 adapter 时忽略
                    pass
        out_frames = list(out.frames[0])
        print(f"[video.sample_v2v] done in {time.time() - t0:.1f}s "
              f"frames={len(out_frames)} seed={seed}", flush=True)
        return {"video": {"frames": out_frames, "fps": out_fps}, "seed": int(seed)}
