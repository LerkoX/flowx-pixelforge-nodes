"""flux-mask-refine 服务端算子注册文件：注册 flux.mask_refine

瓶颈定位打点（dev-log §62，P1）：环境变量 **FLOWX_PROFILE**（在**推理服务进程**上
设置，默认 0 = 关闭、零开销、行为与旧版逐字节一致）：
  1 = 轻量分段计时：CUDA event 总时长 + 每步墙钟 + 反量化入口累计
      （候选逐个 patch）+ transformer.forward 累计；
  2 = 在 1 之上再挂 torch.profiler，打印 top-20 算子表。
报告经 job.progress["profile"] 旁路回传（GET /jobs/{id} 可取，节点侧再打印），
同时写服务端日志。判据：反量化占比高 ⇒ 坐实 Q4 GGUF 逐层反量化为主因。
"""
import os
import random
import time

import torch

from app import execution, preview

FLUX_DIR = os.environ.get("FLUX_DIR", "/models/flux")


def _profile_level():
    """FLOWX_PROFILE 开关：0=关（默认）/ 1=轻量分段 / 2=+torch.profiler。"""
    try:
        return int(os.environ.get("FLOWX_PROFILE", "0") or 0)
    except (TypeError, ValueError):
        return 0


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


class _Profiler:
    """单次执行的计时打点（仅诊断用）。反量化入口用嵌套深度去重：
    内外两层都 patch 时只计最外层，避免双计（不同 diffusers/gguf 版本入口位置不同）。"""

    def __init__(self, level, steps, guide_size):
        self.level = level
        self.steps = steps
        self.guide_size = guide_size
        self.dequant = {"n": 0, "t": 0.0}
        self.forward = {"n": 0, "t": 0.0}
        self.cb = 0.0
        self.walls = []
        self._patched = []
        self._depth = 0
        self._t_last = None
        self._prof = None
        self._table = None

    def _wrap(self, holder, attr, stat):
        orig = getattr(holder, attr, None)
        if orig is None or not callable(orig):
            return False
        prof = self

        def wrapper(*a, **kw):
            if prof._depth > 0:      # 嵌套调用：外层已统计，跳过
                return orig(*a, **kw)
            prof._depth += 1
            t = time.perf_counter()
            try:
                return orig(*a, **kw)
            finally:
                prof._depth -= 1
                stat["n"] += 1
                stat["t"] += time.perf_counter() - t

        try:
            setattr(holder, attr, wrapper)
        except Exception:
            return False
        self._patched.append((holder, attr, orig))
        return True

    def patch(self, model):
        # 反量化入口：gguf 库最内层优先，回退 diffusers 的 GGUFParameter.dequantize
        done = False
        try:
            import gguf.quants as _q
            done = self._wrap(_q, "dequantize", self.dequant)
        except Exception:
            done = False
        if not done:
            try:
                import diffusers.quantizers.gguf.utils as _gu
                cls = getattr(_gu, "GGUFParameter", None)
                if cls is not None:
                    self._wrap(cls, "dequantize", self.dequant)
            except Exception:
                pass
        # transformer 整次 forward（采样时每步调用一次）
        self._wrap(type(model), "forward", self.forward)

    def unpatch(self):
        for holder, attr, orig in self._patched:
            try:
                setattr(holder, attr, orig)
            except Exception:
                pass
        self._patched = []

    def start(self):
        self._t_last = time.perf_counter()

    def tick(self):
        now = time.perf_counter()
        self.walls.append(now - self._t_last if self._t_last is not None else 0.0)
        self._t_last = now

    def profiler(self):
        if self.level < 2:
            return None
        from torch.profiler import ProfilerActivity, profile
        self._prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
        return self._prof

    def close_profiler(self):
        if self._prof is None:
            return
        self._prof.__exit__(None, None, None)
        try:
            self._table = self._prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=20)
        except Exception as e:
            self._table = f"(profiler table failed: {e})"

    def report(self, total_s, crop):
        def pct(x):
            return f"{x / total_s * 100:.1f}%" if total_s else "n/a"
        out = [f"=== flux.mask_refine PROFILE (level={self.level}) ===",
               f"steps={self.steps} guide_size={self.guide_size} crop={crop}",
               f"TOTAL(cuda event) {total_s:.1f}s | "
               f"dequant {self.dequant['t']:.1f}s ({pct(self.dequant['t'])}, "
               f"{self.dequant['n']} calls) | "
               f"forward {self.forward['t']:.1f}s ({pct(self.forward['t'])}, "
               f"{self.forward['n']} calls) | callback {self.cb:.2f}s"]
        for i, w in enumerate(self.walls, 1):
            out.append(f"  step {i}: {w:.1f}s")
        if self._table:
            out.append(self._table)
        return "\n".join(out)


def _fresh_scheduler():
    from diffusers import FlowMatchEulerDiscreteScheduler
    return FlowMatchEulerDiscreteScheduler.from_pretrained(
        os.path.join(FLUX_DIR, "scheduler"))


def _small(img, max_size=512):
    from PIL import Image
    if max(img.size) <= max_size:
        return img
    scale = max_size / max(img.size)
    return img.resize((round(img.width * scale), round(img.height * scale)),
                      Image.Resampling.LANCZOS)


def register(registry):
    @registry.register(
        "flux.mask_refine",
        inputs={
            "model": "MODEL",
            "vae": "VAE",
            "cond": "COND",
            "image": "IMAGE",
            "mask": "IMAGE",
            "padding": "FLOAT",
            "denoise": "FLOAT",
            "steps": "INT",
            "guidance": "FLOAT",
            "seed": "INT",
            "guide_size": "INT",
            "feather": "INT",
            "pipeline_mode": "STRING",
            "attention_slicing": "STRING",
            "cpu_offload": "STRING",
            "controlnet": "MODEL",
            "control_image": "IMAGE",
            "controlnet_scale": "FLOAT",
            "control_mode": "INT",
            "vae_slicing": "BOOL",
            "vae_tiling": "BOOL"
        },
        outputs={"image": "IMAGE"},
        description="FLUX 蒙版局部重绘：根据输入的蒙版外接框外扩裁剪 → 放大到 guide_size → "
                    "FLUX img2img 重绘 → 结合蒙版与边缘羽化无缝贴回原图。"
    )
    def _op_flux_mask_refine(model, vae, cond, image, mask, padding=0.3,
                             denoise=0.8, steps=8, guidance=0.0, seed=-1,
                             guide_size=768, feather=16,
                             pipeline_mode="img2img",
                             attention_slicing="off", cpu_offload="off",
                             vae_slicing=False, vae_tiling=False,
                             controlnet=None, control_image=None,
                             controlnet_scale=0.7, control_mode=2):
        from diffusers import FluxImg2ImgPipeline
        from PIL import Image, ImageDraw, ImageFilter
        import numpy as np

        if type(model).__name__ != "FluxTransformer2DModel":
            raise ValueError("flux.mask_refine: model 需为 flux.unet_load 输出")
        if not isinstance(cond, dict) or "prompt_embeds" not in cond:
            raise ValueError("flux.mask_refine: cond 需为 flux.encode 输出")
        if not torch.cuda.is_available():
            raise ValueError("flux.mask_refine 需要 CUDA")

        seed = int(seed) if int(seed) >= 0 else random.randint(0, 2**32 - 1)
        steps = int(steps)
        job = execution.current()
        rec = preview.recorder_for(job.id, 1) if job else None

        # 确保 mask 为灰度模式（L）且尺寸与原图一致
        mask_l = mask.convert("L")
        if mask_l.size != image.size:
            mask_l = mask_l.resize(image.size, Image.Resampling.LANCZOS)

        # 寻找蒙版外接框 (bbox)
        mask_binary = mask_l.point(lambda p: 255 if p > 128 else 0)
        bbox = mask_binary.getbbox()

        if bbox is None:
            print("[flux.mask_refine] empty mask, no area to refine", flush=True)
            if job:
                job.set_progress(1, 1)
            return {"image": image}

        W, H = image.size
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        px, py = bw * float(padding), bh * float(padding)

        cx1, cy1 = max(0, int(x1 - px)), max(0, int(y1 - py))
        cx2, cy2 = min(W, int(x2 + px)), min(H, int(y2 + py))

        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            print(f"[flux.mask_refine] crop region too small ({cx2-cx1}x{cy2-cy1}), return orig",
                  flush=True)
            if job:
                job.set_progress(1, 1)
            return {"image": image}

        crop_img = image.crop((cx1, cy1, cx2, cy2))
        crop_mask = mask_l.crop((cx1, cy1, cx2, cy2))

        scale = float(guide_size) / min(crop_img.size)
        tw = max(64, round(crop_img.width * scale) // 16 * 16)
        th = max(64, round(crop_img.height * scale) // 16 * 16)
        crop_img_big = crop_img.resize((tw, th), Image.Resampling.LANCZOS)
        crop_mask_big = crop_mask.resize((tw, th), Image.Resampling.NEAREST)

        if rec:
            dbg = image.copy()
            d = ImageDraw.Draw(dbg)
            d.rectangle([cx1, cy1, cx2, cy2], outline=(34, 211, 238), width=3)
            rec.push_pil(_small(dbg), 0.0)

        t0 = time.time()

        level = _profile_level()
        prof = _Profiler(level, steps, guide_size) if level >= 1 else None
        if prof:
            print(f"[flux.mask_refine] PROFILE level={level} ON "
                  f"(steps={steps} guide_size={guide_size})", flush=True)
            prof.patch(model)

        vae_slicing = _as_bool(vae_slicing)
        vae_tiling = _as_bool(vae_tiling)
        offload_mode = str(cpu_offload or "auto").strip().lower()
        pipe_mode = str(pipeline_mode or "img2img").strip().lower()
        # Fill 权重（x_embedder in_channels=384，蒙版+被遮图拼输入）必须走
        # 专属 FluxFillPipeline：FluxInpaintPipeline 按 in_channels//4 推
        # latent 通道数，384//4=96 ≠ VAE 16 通道直接 shape mismatch（exec 1015）
        _tr_in_ch = int(getattr(getattr(model, "config", None),
                                "in_channels", 64) or 64)
        if _tr_in_ch >= 384:
            try:
                from diffusers import FluxFillPipeline
            except ImportError as e:
                raise RuntimeError(
                    "flux.mask_refine: Fill 权重需要 diffusers 的 "
                    "FluxFillPipeline（0.32+），当前环境无该类") from e
            pipe_cls = FluxFillPipeline
            pipe_mode = "fill"
        elif pipe_mode == "inpaint":
            try:
                from diffusers import FluxInpaintPipeline
            except ImportError as e:
                raise RuntimeError(
                    "flux.mask_refine: 当前 diffusers 无 FluxInpaintPipeline，"
                    "无法使用 pipeline_mode=inpaint") from e
            pipe_cls = FluxInpaintPipeline
        elif pipe_mode in ("img2img", "i2i"):
            pipe_mode = "img2img"
            pipe_cls = FluxImg2ImgPipeline
        else:
            raise ValueError(
                f"flux.mask_refine: pipeline_mode 需为 img2img|inpaint，got {pipeline_mode!r}")

        i2i = pipe_cls(
            scheduler=_fresh_scheduler(), vae=vae, text_encoder=None,
            tokenizer=None, text_encoder_2=None, tokenizer_2=None,
            transformer=model)
        i2i.set_progress_bar_config(disable=True)

        # CN 路径不走 accelerate offload → transformer 初始在 CPU，
        # 而 _execution_device 无钩子时回退到 self.device（读 transformer
        # 所在设备）→ 管道会把 prepare 的图像张量放到 CPU，与 GPU 上的 VAE
        # 冲突（exec 1036 设备不匹配报错）。动态子类把执行设备钉在 cuda：
        if controlnet is not None and control_image is not None:
            class _ExecDevCuda(type(i2i)):
                @property
                def _execution_device(self):
                    return torch.device("cuda")
            i2i.__class__ = _ExecDevCuda

        # ---- ControlNet（可选）：全图深度/线稿裁同一 box → VAE encode → pack ----
        # 锁体型轮廓用：denoise=1.0 时 Fill 对蒙版内部完全自由重绘，体型只有
        # Redux 外观锚（压不住围度，exec 1020/1024 实测臀部窄了 ~18%）。
        # CN 残差注入走下方 model.forward 外包，每步 CN 与 transformer 分时上卡。
        cn_enabled = controlnet is not None and control_image is not None
        cn_packed = None
        cn_mode_t = None
        if cn_enabled:
            try:
                print(f"[flux.mask_refine] cn-prep: free="
                      f"{torch.cuda.mem_get_info()[0]/1e9:.1f}GB", flush=True)
                cn_img = control_image.convert("RGB").crop((cx1, cy1, cx2, cy2)) \
                    .resize((tw, th), Image.Resampling.LANCZOS)
                cn_t = torch.from_numpy(
                    np.array(cn_img, dtype=np.float32) / 127.5 - 1.0
                ).permute(2, 0, 1).unsqueeze(0)
                # 模型可能从上轮常驻在卡上，先让位；VAE encode 完即下卡
                model.to("cpu")
                torch.cuda.empty_cache()
                print(f"[flux.mask_refine] cn-prep: model->cpu 后 free="
                      f"{torch.cuda.mem_get_info()[0]/1e9:.1f}GB", flush=True)
                # vae_tiling 设置在下方才生效——CN prep 先按本执行参数显式开
                if _as_bool(vae_tiling):
                    if hasattr(vae, "enable_tiling"): vae.enable_tiling()
                if _as_bool(vae_slicing):
                    if hasattr(vae, "enable_slicing"): vae.enable_slicing()
                vae.to("cuda")
                with torch.no_grad():
                    cl = vae.encode(cn_t.to("cuda", vae.dtype)).latent_dist.mode()
                    cl = (cl - vae.config.shift_factor) * vae.config.scaling_factor
                    print(f"[flux.mask_refine] cn-prep: vae latents "
                          f"{tuple(cl.shape)} free="
                          f"{torch.cuda.mem_get_info()[0]/1e9:.1f}GB", flush=True)
                    cn_packed = i2i._pack_latents(cl, 1, cl.shape[1],
                                                  th // 8, tw // 8)
                # 注意：VAE 必须留在 GPU！Fill 管道的 prepare_latents 紧接着
                # 还要 encode 裁剪图+蒙版图，VAE 在 CPU 上跑 tiled_encode
                # 是 4 核软算几十分钟（exec 1035 卡 74min 的元凶）。
                # transformer 此时已在 CPU，显存富余，VAE 留卡无冲突。
                cn_mode_t = torch.tensor([[int(control_mode)]], device="cuda",
                                         dtype=torch.long)
                print(f"[flux.mask_refine] controlnet ON mode={int(control_mode)} "
                      f"scale={float(controlnet_scale)} cond_packed="
                      f"{tuple(cn_packed.shape)}", flush=True)
            except Exception:
                import traceback
                traceback.print_exc()  # 落容器日志（job error 只带一句）
                raise

        # ---- 显存策略解析 ----
        # auto（推荐默认）：按物理显存自选——装得下就全驻留零搬运，装不下
        # 就走手工时刻表（CN 场景）/ accelerate model offload（非 CN 场景）。
        # dance：CN⇄transformer 每步手工轮换（8GB 卡 CN 场景唯一活路；
        # accelerate 钩子是懒卸载且与 GGUF 量化权重不兼容，exec 1030/1033
        # 两次实测必 OOM）。off：全驻留（大显存卡）。
        if offload_mode == "auto":
            _total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if cn_enabled:
                # 全驻留需求：Fill 6.8 + CN(≤3.3) + VAE 0.4 + 激活余量 ~2.5
                offload_mode = "off" if _total_gb >= 13 else "dance"
            else:
                offload_mode = "off" if _total_gb >= 10 else "model"
            print(f"[flux.mask_refine] cpu_offload=auto → {offload_mode} "
                  f"(vram={_total_gb:.1f}GB)", flush=True)
        if cn_enabled and offload_mode in ("model", "cpu", "on", "true", "1"):
            # CN + accelerate offload 语义统一落到手工 dance（见上注释）
            print(f"[flux.mask_refine] CN 场景 {offload_mode} → dance",
                  flush=True)
            offload_mode = "dance"
        if offload_mode in ("model", "cpu", "on", "true", "1"):
            i2i.enable_model_cpu_offload()
        elif offload_mode in ("sequential", "seq"):
            if cn_enabled:
                raise ValueError("flux.mask_refine: CN 路径不支持 sequential")
            i2i.enable_sequential_cpu_offload()
        elif offload_mode in ("off", "0", "false", "none"):
            model.to("cuda")
            vae.to("cuda")
            torch.cuda.empty_cache()
        elif offload_mode != "dance":
            raise ValueError(f"flux.mask_refine: cpu_offload 需为 auto/dance/"
                             f"model/sequential/off，got {cpu_offload!r}")

        # VAE 切片/分块不改变数学语义，只降低大裁剪区 decode 峰值。
        attn_mode = str(attention_slicing or "off").strip().lower()
        if attn_mode not in ("", "off", "none", "0", "false"):
            i2i.enable_attention_slicing("auto" if attn_mode == "auto" else attn_mode)
        if vae_slicing:
            if hasattr(i2i, "enable_vae_slicing"):
                i2i.enable_vae_slicing()
            elif hasattr(i2i.vae, "enable_slicing"):
                i2i.vae.enable_slicing()
        if vae_tiling:
            if hasattr(i2i, "enable_vae_tiling"):
                i2i.enable_vae_tiling()
            elif hasattr(i2i.vae, "enable_tiling"):
                i2i.vae.enable_tiling()
        print(f"[flux.mask_refine] crop -> {tw}x{th} guide={guide_size} "
              f"steps={steps} denoise={denoise} mode={pipe_mode} "
              f"attn={attn_mode} cpu_offload={offload_mode} "
              f"vae_slice={vae_slicing} vae_tile={vae_tiling}", flush=True)

        # ---- ControlNet 残差注入：外包 transformer.forward ----
        # 每步：CN 上卡前向（offload 模式下 transformer 在 CPU/meta，不撞显存）
        # → CN 下卡 → 原 forward（offload 钩子上卡 transformer）。Fill 的
        # hidden_states 前 64 通道即含噪 latents（diffusers concat 顺序
        # [latents64, masked_img64, mask256]），切出来喂 CN；非 Fill 直接整体。
        orig_fwd = None
        if cn_enabled:
            _dance = (offload_mode == "dance")
            orig_fwd = model.forward  # 纯 forward（CN 路径未挂 accelerate 钩子）
            _cn = controlnet
            # int8 量化后首参可能是 QTensor（dtype=int8），必须挑浮点参定计算精度
            _cn_dtype = next((p.dtype for p in _cn.parameters()
                              if p.is_floating_point()), torch.float16)
            _cn_scale = float(controlnet_scale)
            _guidance_val = float(guidance)
            _total_steps = int(steps)
            _step_no = [0]
            if _dance:
                # 手工时刻表初始态：transformer 从 CPU 起步（CN 先行）
                model.to("cpu")
            else:
                # 全驻留：CN 一次性上卡，wrapper 不再做设备搬运
                _cn.to("cuda")
                torch.cuda.empty_cache()

            def _fwd_with_cn(*args, **kwargs):
                hs = kwargs.get("hidden_states")
                if hs is None and args:
                    hs = args[0]
                noisy = hs[..., :64] if hs.shape[-1] >= 384 else hs
                # Fill 模型的 config guidance_embeds=false（dev 底 config 遗留，
                # guidance_embedder 权重未用）→ 管道不传 guidance；但 CN 的
                # config guidance_embeds=true，time_text_embed 必吃三项
                # → 缺省时用节点 guidance 参数合成
                g = kwargs.get("guidance")
                if g is None:
                    g = torch.full((hs.shape[0],), _guidance_val,
                                   device="cuda", dtype=torch.float32)
                _step_no[0] += 1
                if _dance:
                    # 手工时刻表（每步）：transformer 下卡 → CN 上卡算残差 →
                    # CN 下卡 → transformer 上卡跑本体 → 留在卡上（下步开头
                    # 再挪）。PCIe 来回 ~4s/步，换 8GB 卡确定性不 OOM。
                    # 第 1 步时 prepare 相位（VAE encode）已结束：VAE 下卡腾地方
                    if _step_no[0] == 1:
                        try:
                            if next(vae.parameters()).device.type == "cuda":
                                vae.to("cpu")
                        except Exception:
                            pass
                    if next(model.parameters()).device.type == "cuda":
                        model.to("cpu")
                        torch.cuda.empty_cache()
                    _cn.to("cuda")
                with torch.no_grad():
                    cn_bs, cn_ss = _cn(
                        hidden_states=noisy.to(_cn_dtype),
                        controlnet_cond=cn_packed.to("cuda", _cn_dtype),
                        controlnet_mode=cn_mode_t,
                        conditioning_scale=_cn_scale,
                        encoder_hidden_states=kwargs["encoder_hidden_states"].to(_cn_dtype),
                        pooled_projections=kwargs["pooled_projections"].to(_cn_dtype),
                        timestep=kwargs["timestep"],
                        img_ids=kwargs["img_ids"],
                        txt_ids=kwargs["txt_ids"],
                        guidance=g,
                        return_dict=False)
                if _dance:
                    _cn.to("cpu")
                    torch.cuda.empty_cache()
                    model.to("cuda")
                kwargs["controlnet_block_samples"] = \
                    [t.to(hs.dtype) for t in cn_bs]
                kwargs["controlnet_single_block_samples"] = \
                    [t.to(hs.dtype) for t in cn_ss]
                out = orig_fwd(*args, **kwargs)
                # 末步（仅 dance）：transformer 下卡、VAE 回卡准备 decode
                if _dance and _step_no[0] >= _total_steps:
                    model.to("cpu")
                    torch.cuda.empty_cache()
                    try:
                        vae.to("cuda")
                    except Exception:
                        pass
                return out

            model.forward = _fwd_with_cn

        pe = cond["prompt_embeds"].to("cuda", model.dtype)
        pooled = cond["pooled"].to("cuda", model.dtype)

        def _cb(p, si, t, kw):
            execution.check_cancelled()
            if job:
                job.set_progress(si + 1, steps)
            if prof:
                prof.tick()
            return kw

        i2i_prof = prof.profiler() if prof else None
        if i2i_prof:
            i2i_prof.__enter__()
        ev0 = torch.cuda.Event(enable_timing=True) if prof else None
        if ev0:
            ev0.record()
        if prof:
            prof.start()
        call_kw = dict(
            prompt_embeds=pe, pooled_prompt_embeds=pooled,
            image=crop_img_big, strength=float(denoise),
            # 必传！缺省 height/width 时管道按 default_sample_size(128)*8=1024
            # 跑方图，蒙版条件被横向压扁→模型按“胖子”生成→贴回纵向拉伸后
            # 体型窄 ~18%（exec 1020/1024）；同时 CN cond 与 latents 尺寸错位
            # （exec 1027 的 4096 vs 2400）。传入裁剪区真实尺寸后两者皆愈。
            height=th, width=tw,
            num_inference_steps=steps, guidance_scale=float(guidance),
            generator=torch.Generator("cpu").manual_seed(seed),
            callback_on_step_end=_cb,
            callback_on_step_end_tensor_inputs=[],
        )
        if pipe_mode in ("inpaint", "fill"):
            call_kw["mask_image"] = crop_mask_big
        try:
            refined_big = i2i(**call_kw).images[0]
        except Exception:
            import traceback
            traceback.print_exc()  # 落容器日志，便于定位（job error 只带一句）
            raise
        finally:
            if orig_fwd is not None:
                model.forward = orig_fwd
            # 清干净 accelerate offload 钩子：不清理则异常中断时 transformer
            # 卡在 GPU 上且带钩子（后绩 .to() 静默失效，下轮 cn-prep OOM）
            try:
                i2i.maybe_free_model_hooks()
            except Exception:
                pass
        gpu_total = 0.0
        if ev0:
            ev1 = torch.cuda.Event(enable_timing=True)
            ev1.record()
            torch.cuda.synchronize()
            gpu_total = ev0.elapsed_time(ev1) / 1000.0
        if prof:
            prof.close_profiler()

        refined = refined_big.resize(crop_img.size, Image.Resampling.LANCZOS)

        feather_val = int(feather)
        border_mask = Image.new("L", crop_img.size, 0)
        if feather_val > 0 and crop_img.width > 2 * feather_val and crop_img.height > 2 * feather_val:
            ImageDraw.Draw(border_mask).rectangle(
                [feather_val, feather_val, crop_img.width - feather_val, crop_img.height - feather_val],
                fill=255
            )
            border_mask = border_mask.filter(ImageFilter.GaussianBlur(feather_val / 2))
        else:
            border_mask = Image.new("L", crop_img.size, 255)

        m1 = np.array(crop_mask, dtype=np.float32) / 255.0
        m2 = np.array(border_mask, dtype=np.float32) / 255.0
        paste_m = np.clip(m1 * m2 * 255.0, 0, 255).astype(np.uint8)
        paste_mask = Image.fromarray(paste_m, mode="L")

        out_img = image.copy()
        out_img.paste(refined, (cx1, cy1), paste_mask)

        if job:
            job.set_progress(1, 1)
        if rec:
            rec.push_pil(_small(out_img), 1.0)

        print(f"[flux.mask_refine] crop ({cx1},{cy1},{cx2},{cy2}) -> {tw}x{th} "
              f"denoise={denoise} steps={steps} ({time.time()-t0:.1f}s)",
              flush=True)

        if prof:
            rep = prof.report(gpu_total, f"{tw}x{th}")
            print(rep, flush=True)
            prof.unpatch()
            if job:                 # 最后塞入（set_progress 会重置 progress dict）
                job.progress["profile"] = rep

        return {"image": out_img}
