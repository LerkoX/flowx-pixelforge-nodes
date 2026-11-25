"""flashvsr 族：FlashVSR v1.1 视频超分（官方 diffsynth 管线包装）。

算子：
- video.flashvsr(video, scale, sparse_ratio, local_range, seed, color_fix)
  → video（超分后）+ fps/帧数/宽高

背景：ComfyUI 生态 2025 后的视频高清化主流是专用视频超分模型
（SeedVR2 / FlashVSR），FlashVSR 是**一步蒸馏**的流式视频 SR
（arXiv 2510.12747），基于 Wan2.1 VAE/DiT——与我们栈亲和。官方
（OpenImagingLab/FlashVSR）给出 4× 推荐设置，A100 上 768×1408 ≈17fps。

依赖（机器一次性安装，见节点 README）：
- diffsynth 包：官方仓库 `pip install -e . --no-deps`（FLASHVSR_REPO
  默认 /opt/flashvsr；**绝不装 requirements.txt**——它 pin torch 2.6
  会降级我们的 torch 2.13）
- block_sparse_attn：wheel 或源码编译（mit-han-lab/Block-Sparse-
  Attention，sm_80+；**无 → 明确报错，不做 dense 降级**——4× 输出下
  seqlen² 的 dense 掩码显存不可行，1920×3328 时 ~2.5GB/层且官方实测
  无 LCSA 质量明显下降）
- 权重 FLASHVSR_DIR（默认 /var/lib/flowx-inference/models/flashvsr）：
  LQ_proj_in.ckpt / TCDecoder.ckpt / Wan2.1_VAE.pth /
  diffusion_pytorch_model_streaming_dmd.safetensors / posi_prompt.pth
  （posi_prompt.pth 在官方 git 仓库 examples/WanVSR/prompt_tensor/）

帧契约：8n+1（largest_8n1_leq，尾帧 pad 4）；输出尺寸 = scale× 输入
向下取 128 倍数（480×832 4× = 1920×3328，恰好都是 128 倍数）。
"""
import os
import time

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
OP_VERSION = "213c6eccb1a3"
OP_FAMILY = "flashvsr"  # 必须等于本文件 stem（gate J 校验）

class _VersionedRegistry:
    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

FLASHVSR_DIR = os.environ.get(
    "FLASHVSR_DIR", "/var/lib/flowx-inference/models/flashvsr")
FLASHVSR_REPO = os.environ.get("FLASHVSR_REPO", "/opt/flashvsr")


def _models():
    from app import model_manager
    return model_manager


def _video_to_frames(video_data, max_frames=0):
    """VIDEO data → (frames PIL 列表, fps)。path 形态走 ffmpeg 裸流解码。"""
    if isinstance(video_data, dict) and video_data.get("frames"):
        frames = video_data["frames"]
        if max_frames:
            frames = frames[:int(max_frames)]
        return frames, float(video_data.get("fps") or 24.0)
    path = (video_data or {}).get("path")
    if not path or not os.path.isfile(path):
        raise ValueError(f"video.flashvsr: 视频对象无 frames 且 path 无效: "
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
    num, _, den = rate_s.partition("/")
    fps = float(num) / float(den) if den and float(den) else float(num)
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if max_frames:
        cmd += ["-frames:v", str(int(max_frames))]
    cmd += ["-f", "image2pipe", "-pix_fmt", "rgb24", "-vcodec", "rawvideo", "-"]
    width, height = int(w_s), int(h_s)
    raw = subprocess.run(cmd, capture_output=True, timeout=1800).stdout
    import numpy as np
    from PIL import Image
    fsz = width * height * 3
    n = len(raw) // fsz
    arr = np.frombuffer(raw[:n * fsz], dtype=np.uint8).reshape(
        (n, height, width, 3))
    return [Image.fromarray(arr[i]) for i in range(n)], fps


def _check_weights():
    need = ("LQ_proj_in.ckpt", "TCDecoder.ckpt", "Wan2.1_VAE.pth",
            "diffusion_pytorch_model_streaming_dmd.safetensors",
            "posi_prompt.pth")
    missing = [f for f in need
               if not os.path.exists(os.path.join(FLASHVSR_DIR, f))]
    if missing:
        raise FileNotFoundError(
            f"video.flashvsr: 缺权重 {missing}（{FLASHVSR_DIR}）——"
            f"先查共享盘 /model，没有再 hf-mirror 下 JunhaoZhuang/FlashVSR-v1.1，"
            f"posi_prompt.pth 在官方 git 仓库 examples/WanVSR/prompt_tensor/")


def _load_pipe():
    """FlashVSR 管线常驻（DiT 1.3B + TCDecoder + LQ_proj，~2GB 权重）。"""
    key = "flashvsr:v1.1"
    models = _models()
    if key in models._pipes:
        return models.get(key)
    _check_weights()
    import sys
    import torch
    if FLASHVSR_REPO not in sys.path:
        sys.path.insert(0, FLASHVSR_REPO)
    try:
        import block_sparse_attn  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "video.flashvsr: 缺 block_sparse_attn（LCSA 必需，官方实测无它"
            "质量明显下降；dense 降级在 4× 输出下显存不可行）——装 wheel 或"
            "编译 mit-han-lab/Block-Sparse-Attention") from e
    t0 = time.time()
    from diffsynth import ModelManager, FlashVSRTinyLongPipeline
    from utils.utils import Causal_LQ4x_Proj
    from utils.TCDecoder import build_tcdecoder
    mm = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
    mm.load_models([os.path.join(
        FLASHVSR_DIR, "diffusion_pytorch_model_streaming_dmd.safetensors")])
    pipe = FlashVSRTinyLongPipeline.from_model_manager(mm, device="cuda")
    dit = pipe.denoising_model()
    dit.LQ_proj_in = Causal_LQ4x_Proj(
        in_dim=3, out_dim=1536, layer_num=1).to("cuda", dtype=torch.bfloat16)
    dit.LQ_proj_in.load_state_dict(
        torch.load(os.path.join(FLASHVSR_DIR, "LQ_proj_in.ckpt"),
                   map_location="cpu"), strict=True)
    dit.LQ_proj_in.to("cuda")
    pipe.TCDecoder = build_tcdecoder(
        new_channels=[512, 256, 128, 128], new_latent_channels=16 + 768)
    pipe.TCDecoder.load_state_dict(
        torch.load(os.path.join(FLASHVSR_DIR, "TCDecoder.ckpt"),
                   map_location="cpu"), strict=False)
    pipe.to("cuda")
    pipe.enable_vram_management(num_persistent_param_in_dit=None)
    # init_cross_kv 默认读仓库相对路径 posi_prompt.pth——显式传权重目录版
    ctx = torch.load(os.path.join(FLASHVSR_DIR, "posi_prompt.pth"),
                     map_location="cuda")
    pipe.init_cross_kv(context_tensor=ctx)
    pipe.load_models_to_device(["dit", "vae"])
    est = sum(os.path.getsize(os.path.join(FLASHVSR_DIR, f))
              for f in ("diffusion_pytorch_model_streaming_dmd.safetensors",
                        "LQ_proj_in.ckpt", "TCDecoder.ckpt", "Wan2.1_VAE.pth")
              if os.path.isfile(os.path.join(FLASHVSR_DIR, f)))
    with models._lock:
        models._evict_if_needed(est)
        models._pipes[key] = pipe
        models._archs[key] = "flashvsr"
        models._sizes[key] = est
        models._last_used[key] = time.time()
    print(f"[flashvsr] pipe loaded {time.time()-t0:.1f}s "
          f"est={est/1e9:.1f}GB", flush=True)
    return models.get(key)


def _largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def _prep_dims(w0, h0, scale, multiple=128):
    sW, sH = int(round(w0 * scale)), int(round(h0 * scale))
    tW, tH = (sW // multiple) * multiple, (sH // multiple) * multiple
    if tW == 0 or tH == 0:
        raise ValueError(f"放大后 {sW}x{sH} 不足 128 倍数（scale={scale}）")
    return sW, sH, tW, tH


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)

    def _op_flashvsr(video, scale=4.0, sparse_ratio=2.0, local_range=11,
                     seed=0, color_fix=1, max_frames=0):
        import numpy as np
        import torch
        from PIL import Image

        t0 = time.time()
        frames, fps = _video_to_frames(video, int(max_frames or 0))
        n0 = len(frames)
        w0, h0 = frames[0].size
        scale = float(scale)
        sW, sH, tW, tH = _prep_dims(w0, h0, scale)
        print(f"[flashvsr] {n0} 帧 {w0}x{h0} → x{scale} {sW}x{sH} "
              f"→ 128倍数 {tW}x{tH}", flush=True)

        # 帧契约 8n+1（尾帧 pad 4 后 largest_8n1_leq）
        idx = list(range(n0)) + [n0 - 1] * 4
        F = _largest_8n1_leq(len(idx))
        if F == 0:
            raise ValueError(f"帧数不足（{n0} 帧）")
        idx = idx[:F]
        print(f"[flashvsr] 帧契约 {n0}→{F}（8n+1）", flush=True)

        # LQ 输入：bicubic 放大到 scale× 后 center-crop 到 128 倍数
        #（官方 prepare_input_tensor 同款语义）
        lq = []
        for i in idx:
            img = frames[i]
            up = img.resize((sW, sH), Image.BICUBIC)
            l, t = (sW - tW) // 2, (sH - tH) // 2
            t_arr = torch.from_numpy(
                np.asarray(up.crop((l, t, l + tW, t + tH)), np.uint8)
            ).permute(2, 0, 1).float().div(255.0).mul_(2.0).sub_(1.0)
            lq.append(t_arr.to(torch.bfloat16))
        LQ = torch.stack(lq, 0).permute(1, 0, 2, 3).unsqueeze(0)  # 1 C F H W
        del lq
        print(f"[flashvsr] LQ 张量 {tuple(LQ.shape)} 就位 "
              f"({time.time()-t0:.0f}s)", flush=True)

        pipe = _load_pipe()
        topk = float(sparse_ratio) * 768 * 1280 / (tH * tW)
        t1 = time.time()
        out = pipe(
            prompt="", negative_prompt="", cfg_scale=1.0,
            num_inference_steps=1, seed=int(seed),
            LQ_video=LQ, num_frames=F, height=tH, width=tW,
            is_full_block=False, if_buffer=True,
            topk_ratio=topk, kv_ratio=3.0,
            local_range=int(local_range),
            color_fix=bool(int(color_fix)),
        )
        print(f"[flashvsr] 推理 {time.time()-t1:.1f}s", flush=True)
        del LQ
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from einops import rearrange
        arr = rearrange(out, "C T H W -> T H W C")
        arr = ((arr.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(
            np.uint8)
        out_frames = [Image.fromarray(arr[i]) for i in range(arr.shape[0])]
        print(f"[flashvsr] done {time.time()-t0:.1f}s: {n0} 帧 "
              f"{w0}x{h0} → {len(out_frames)} 帧 {out_frames[0].size}",
              flush=True)
        return {"video": {"frames": out_frames, "fps": fps},
                "fps": int(round(fps)), "num_frames": len(out_frames),
                "width": out_frames[0].size[0],
                "height": out_frames[0].size[1]}

    registry.register(
        "video.flashvsr",
        {"video": "VIDEO", "scale": "FLOAT", "sparse_ratio": "FLOAT",
         "local_range": "INT", "seed": "INT", "color_fix": "INT",
         "max_frames": "INT"},
        {"video": "VIDEO", "fps": "INT", "num_frames": "INT",
         "width": "INT", "height": "INT"},
        description="FlashVSR v1.1 视频超分（一步蒸馏流式 SR，官方推荐 4×；"
                    "480×832→1920×3328。需 block_sparse_attn + 权重五件套）")
