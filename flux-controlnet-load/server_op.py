"""flux-controlnet-load 节点的服务端算子插件（随节点包自注册，不动服务端核心）。

装载 FLUX ControlNet（Shakker-Labs Union-Pro：depth/canny/tile 等多模态合一，
FluxControlNetModel 原生 safetensors），注册进 model_manager LRU
（键 flux-cn:<目录名>），与 flux.unet_load 的常驻/淘汰语义一致。

权重布局：/models/flux/<path>/ 下需有 config.json + diffusion_pytorch_model.safetensors
（如 cn-union-pro/）。Union-Pro-2.0 原盘为 BF16 6.6GB（~3.3B 参数），
装载即转 fp16（逐张量，无全量双缓冲），int8 量化后 ~3.6GB 常驻。
教训：不要按文件原生 bf16 装载再 int8 量化（exec 1061 蒙版区全黑）。
装载峰值靠流水线排布解决：CNLoad 须排在 UnetLoad 之前。

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


def _header_dtypes(st_path):
    """读 safetensors header，返回 (dtype 集合, 参数量)；失败返回 (set(), 0)。"""
    import json
    import struct
    dtypes, params = set(), 0
    try:
        with open(st_path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(min(n, 64 << 20)))
        for k, v in hdr.items():
            if k == "__metadata__":
                continue
            dtypes.add(v.get("dtype", ""))
            p = 1
            for dim in v.get("shape", []):
                p *= int(dim)
            params += p
    except Exception:
        pass
    return dtypes, params


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
        file_dts, params = _header_dtypes(st)
        # 始终按请求 dtype（默认 fp16）装载：bf16 原盘会在装载时逐张量转
        # fp16（无全量双缓冲）。不要为省峰值改走 bf16 原生装载再量化——
        # exec 1061 实证该路径产出的 int8 CN 参与采样后蒙版区全黑（玩具
        # Linear 实验测不出，真实 FluxControlNetModel 上炸）；装载峰值应由
        # 流水线排布解决（CNLoad 排在 UnetLoad 前，6.6+1.5GB 足够安全）。
        load_dt = _DTYPES[dt]
        cn = FluxControlNetModel.from_pretrained(d, torch_dtype=load_dt)
        # 常驻估算按参数量推导（原公式 est=size*0.27 误把 bf16 原盘当 fp32，
        # 实估 1.7GB 而真实 int8 常驻 ~3.6GB——低估 2x 会让 LRU 超收，
        # 16GB 盒上直接 OOM）。int8≈1B/参+scale 元数据，fp16=2B/参。
        est = int(params * 2) if params else os.path.getsize(st)
        if qmode == "int8":
            try:
                from optimum.quanto import freeze, qint8, quantize as _q
                _q(cn, weights=qint8)
                freeze(cn)
                print("[flux.controlnet_load] int8 量化完成", flush=True)
                est = int(params * 1.1) if params else int(est * 0.55)
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
