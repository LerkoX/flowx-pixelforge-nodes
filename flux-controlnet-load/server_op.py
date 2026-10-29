"""flux-controlnet-load 节点的服务端算子插件（随节点包自注册，不动服务端核心）。

装载 FLUX ControlNet（Shakker-Labs Union-Pro：depth/canny/tile 等多模态合一，
FluxControlNetModel 原生 safetensors），注册进 model_manager LRU
（键 flux-cn:<目录名>），与 flux.unet_load 的常驻/淘汰语义一致。

权重布局：/models/flux/<path>/ 下需有 config.json + diffusion_pytorch_model.safetensors
（如 cn-union-pro/）。fp32 原盘 6.6GB，装载即转 fp16（~3.3GB 常驻）。

用途：配合 flux-mask-refine 的 controlnet 输入，在 Fill 局部重绘时注入
深度/线稿结构约束（锁定体型轮廓，防止 denoise=1.0 自由重绘走形）。

算子：
- flux.controlnet_load(path, dtype) → CONTROLNET（FluxControlNetModel 引用）
"""
import os
import time

import torch

FLUX_DIR = os.environ.get("FLUX_DIR", "/models/flux")

_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}


def _models():
    from app.main import models  # 惰性导入：插件扫描时 app.main 尚在初始化
    return models


def register(registry):
    @registry.register(
        "flux.controlnet_load",
        inputs={"path": "STRING", "dtype": "STRING", "quantize": "STRING"},
        outputs={"controlnet": "MODEL", "info": "STRING"},
        description="FLUX ControlNet 加载（Union-Pro 多模态：depth=2/canny=0/"
                    "tile=1/pose=4 等）：/models/flux/<path>/ 目录装载为 fp16，"
                    "可再 int8 量化（16GB 内存机活命槽：3.3GB→1.7GB 常驻），"
                    "注册进 model_manager LRU（键 flux-cn:<path>:<量化>），"
                    "供 flux.mask_refine 的 controlnet 输入使用")
    def _op_flux_controlnet_load(path="cn-union-pro", dtype="fp16",
                                 quantize="int8"):
        from diffusers import FluxControlNetModel

        dt = str(dtype or "fp16").strip().lower()
        if dt not in _DTYPES:
            raise ValueError(f"flux.controlnet_load: dtype 需为 "
                             f"{sorted(_DTYPES)}，got {dtype!r}")
        qmode = str(quantize or "int8").strip().lower()
        if qmode in ("none", "fp16", "off", "0", "false"):
            qmode = "off"
        if qmode not in ("int8", "off"):
            raise ValueError(f"flux.controlnet_load: quantize 需为 int8/off，"
                             f"got {quantize!r}")
        key = f"flux-cn:{path}:{qmode}"
        models = _models()
        if key in models._pipes:
            return {"controlnet": models.get(key),
                    "info": f"cached {key}"}

        d = os.path.join(FLUX_DIR, path)
        cfg = os.path.join(d, "config.json")
        st = os.path.join(d, "diffusion_pytorch_model.safetensors")
        if not os.path.isfile(cfg) or not os.path.isfile(st):
            raise ValueError(
                f"flux.controlnet_load: {d}/ 下需有 config.json + "
                f"diffusion_pytorch_model.safetensors（当前缺失）")
        t0 = time.time()
        cn = FluxControlNetModel.from_pretrained(d, torch_dtype=_DTYPES[dt])
        # int8 量化（默认开）：16GB 内存机的活命槽，fp16 3.3GB → int8
        # ~1.7GB 常驻。采样期 RAM = Fill 6.8 + CN + 服务 1.5，fp16 时
        # 11.6GB 顶格+页缓存即 OOM Kill 容器（exec 1036 实死）；
        # int8 后 ~10GB 有喘息。quanto 不支持的层自动退回 fp16；
        # Pascal 上走 dequant+fp16 mm 路径。
        est = os.path.getsize(st) // 2  # fp32 原盘 → fp16 常驻估算
        if qmode == "int8":
            try:
                from optimum.quanto import freeze, qint8, quantize as _q
                _q(cn, weights=qint8)
                freeze(cn)
                print("[flux.controlnet_load] int8 量化完成", flush=True)
                est = int(os.path.getsize(st) * 0.27)  # int8≈fp16 的 52%+scale
            except Exception as e:
                print(f"[flux.controlnet_load] int8 量化失败，退回 fp16: {e}",
                      flush=True)
        with models._lock:
            models._evict_if_needed(est)
            models._pipes[key] = cn
            models._archs[key] = "FluxControlNetModel"
            models._sizes[key] = est
            models._last_used[key] = time.time()
        info = (f"{path} dtype={dt} quantize={qmode} "
                f"est={est/1e9:.1f}GB ({time.time()-t0:.1f}s)")
        print(f"[flux.controlnet_load] registered '{key}' {info}", flush=True)
        return {"controlnet": cn, "info": info}
