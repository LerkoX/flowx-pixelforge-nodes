"""FLUX PuLID 人脸身份编码（PuLID-FLUX v0.9.1 移植，随节点包自注册）。

机制（对拍 InstantX/PuLID pulid/pipeline_flux.py get_id_embedding）：
- insightface antelopev2 取最大脸 512 维 ArcFace 特征（onnxruntime CPU）
- facexlib retinaface 检测 + 5 点对齐裁 512x512 → bisenet 解析抠出脸部
  （背景置白、面部转灰——只保留面部结构特征，防肤色/妆容泄漏）
- EVA-CLIP（EVA02-CLIP-L-14-336，自定义 fork）编码得 cls 特征(768) +
  5 层中间 hidden（idx 4/8/12/16/20，各 (N,1024)）
- IDFormer（perceiver resampler）融合 [ante512 + eva768] 与 vit hidden
  → id_embedding (1, 32, 2048)
- FACE 对象 = {"id_embedding", "ca": 20×PerceiverAttentionCA}——CA 模块随
  对象走，采样侧（flux.sample 的 pulid 输入）前向 hook 注入，本插件不碰
  transformer。

模型资产（PULID_DIR，默认 /models/pulid，需预先放置）：
- pulid_flux_v0.9.1.safetensors（guozinan/PuLID，IDFormer+CA 权重 ~1.1GB）
- EVA02_CLIP_L_336_psz14_s6B.pt（QuanSun/EVA-CLIP ~1.7GB）
- facexlib/detection_Resnet50_Final.pth + parsing_bisenet.pth
- lib/facexlib + lib/eva_clip（源码包，sys.path 惰性挂载，运行期零网络）
- insightface antelopev2 沿用 INSIGHTFACE_HOME（/models/insightface）

显存相位（GTX1080 8GB）：编码前 transformer 全部搬回 CPU → EVA(bf16~1.2GB)
+解析+IDFormer 上卡编码 → 全部回 CPU 驻留内存（下次秒回）。

License 注意：PuLID 权重官方声明仅限研究/非商用。
"""
import os
import sys
import time

import torch

PULID_DIR = os.environ.get("PULID_DIR", "/models/pulid")
PULID_LIB = os.path.join(PULID_DIR, "lib")


def _models():
    from app.main import models
    return models


def _move_transformers(device):
    """把所有已注册 FluxTransformer2DModel 搬到指定设备（编码相位腾 VRAM）。"""
    n = 0
    for p in list(_models()._pipes.values()):
        if type(p).__name__ == "FluxTransformer2DModel":
            p.to(device)
            n += 1
    if n:
        torch.cuda.empty_cache()
        print(f"[flux.pulid_encode] {n} transformer -> {device}", flush=True)


# ---------------------------------------------------------------------------
# PuLID 模块定义（移植自 ToTheBeginning/PuLID pulid/encoders_transformer.py，
# 保持键名与官方权重严格一致；惰性构造以兼容无 GPU 契约测试的导入）
# ---------------------------------------------------------------------------
_PULID_CLS = None


def _pulid_classes():
    global _PULID_CLS
    if _PULID_CLS is not None:
        return _PULID_CLS
    import math
    from torch import nn

    def _feed_forward(dim, mult=4):
        inner_dim = int(dim * mult)
        return nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim, bias=False),
            nn.GELU(),
            nn.Linear(inner_dim, dim, bias=False),
        )

    def _reshape_tensor(x, heads):
        bs, length, width = x.shape
        x = x.view(bs, length, heads, -1)
        x = x.transpose(1, 2)
        x = x.reshape(bs, heads, length, -1)
        return x

    class PerceiverAttentionCA(nn.Module):
        def __init__(self, *, dim=3072, dim_head=128, heads=16, kv_dim=2048):
            super().__init__()
            self.scale = dim_head ** -0.5
            self.dim_head = dim_head
            self.heads = heads
            inner_dim = dim_head * heads
            self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
            self.norm2 = nn.LayerNorm(dim)
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim,
                                   inner_dim * 2, bias=False)
            self.to_out = nn.Linear(inner_dim, dim, bias=False)

        def forward(self, x, latents):
            x = self.norm1(x)
            latents = self.norm2(latents)
            b, seq_len, _ = latents.shape
            q = self.to_q(latents)
            k, v = self.to_kv(x).chunk(2, dim=-1)
            q = _reshape_tensor(q, self.heads)
            k = _reshape_tensor(k, self.heads)
            v = _reshape_tensor(v, self.heads)
            scale = 1 / math.sqrt(math.sqrt(self.dim_head))
            weight = (q * scale) @ (k * scale).transpose(-2, -1)
            weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
            out = weight @ v
            out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)
            return self.to_out(out)

    class PerceiverAttention(nn.Module):
        def __init__(self, *, dim, dim_head=64, heads=8, kv_dim=None):
            super().__init__()
            self.scale = dim_head ** -0.5
            self.dim_head = dim_head
            self.heads = heads
            inner_dim = dim_head * heads
            self.norm1 = nn.LayerNorm(dim if kv_dim is None else kv_dim)
            self.norm2 = nn.LayerNorm(dim)
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(dim if kv_dim is None else kv_dim,
                                   inner_dim * 2, bias=False)
            self.to_out = nn.Linear(inner_dim, dim, bias=False)

        def forward(self, x, latents):
            x = self.norm1(x)
            latents = self.norm2(latents)
            b, seq_len, _ = latents.shape
            q = self.to_q(latents)
            kv_input = torch.cat((x, latents), dim=-2)
            k, v = self.to_kv(kv_input).chunk(2, dim=-1)
            q = _reshape_tensor(q, self.heads)
            k = _reshape_tensor(k, self.heads)
            v = _reshape_tensor(v, self.heads)
            scale = 1 / math.sqrt(math.sqrt(self.dim_head))
            weight = (q * scale) @ (k * scale).transpose(-2, -1)
            weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
            out = weight @ v
            out = out.permute(0, 2, 1, 3).reshape(b, seq_len, -1)
            return self.to_out(out)

    class IDFormer(nn.Module):
        """perceiver resampler 式 ID 编码器：id embedding(arcface) 与 query
        token 拼成 latents，逐层与 5 个尺度的 vit 特征交叉注意力。"""

        def __init__(self, dim=1024, depth=10, dim_head=64, heads=16,
                     num_id_token=5, num_queries=32, output_dim=2048,
                     ff_mult=4):
            super().__init__()
            self.num_id_token = num_id_token
            self.dim = dim
            self.num_queries = num_queries
            assert depth % 5 == 0
            self.depth = depth // 5
            scale = dim ** -0.5
            self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * scale)
            self.proj_out = nn.Parameter(scale * torch.randn(dim, output_dim))
            self.layers = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                        PerceiverAttention(dim=dim, dim_head=dim_head,
                                           heads=heads),
                        _feed_forward(dim=dim, mult=ff_mult),
                    ])
                )
            for i in range(5):
                setattr(self, f"mapping_{i}", nn.Sequential(
                    nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(),
                    nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(),
                    nn.Linear(1024, dim),
                ))
            self.id_embedding_mapping = nn.Sequential(
                nn.Linear(1280, 1024), nn.LayerNorm(1024), nn.LeakyReLU(),
                nn.Linear(1024, 1024), nn.LayerNorm(1024), nn.LeakyReLU(),
                nn.Linear(1024, dim * num_id_token),
            )

        def forward(self, x, y):
            latents = self.latents.repeat(x.size(0), 1, 1)
            num_duotu = x.shape[1] if x.ndim == 3 else 1
            x = self.id_embedding_mapping(x)
            x = x.reshape(-1, self.num_id_token * num_duotu, self.dim)
            latents = torch.cat((latents, x), dim=1)
            for i in range(5):
                vit_feature = getattr(self, f"mapping_{i}")(y[i])
                ctx_feature = torch.cat((x, vit_feature), dim=1)
                for attn, ff in self.layers[i * self.depth:(i + 1) * self.depth]:
                    latents = attn(ctx_feature, latents) + latents
                    latents = ff(latents) + latents
            latents = latents[:, :self.num_queries]
            latents = latents @ self.proj_out
            return latents

    _PULID_CLS = (PerceiverAttentionCA, PerceiverAttention, IDFormer)
    return _PULID_CLS


# ---------------------------------------------------------------------------
# 惰性资源单例
# ---------------------------------------------------------------------------
_STATE = {}


def _ensure_lib():
    if PULID_LIB not in sys.path:
        if not os.path.isdir(PULID_LIB):
            raise FileNotFoundError(
                f"PuLID 依赖库目录不存在：{PULID_LIB}（需预置 facexlib 与 "
                "eva_clip 源码包，见插件 docstring）")
        sys.path.insert(0, PULID_LIB)


def _load_adapter():
    """IDFormer + N×PerceiverAttentionCA（CPU 驻留，用时上卡）。"""
    if "adapter" in _STATE:
        return _STATE["adapter"]
    from safetensors.torch import load_file
    path = os.path.join(PULID_DIR, "pulid_flux_v0.9.1.safetensors")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"PuLID 适配器权重缺失：{path}")
    CA, _, IDFormer = _pulid_classes()
    sd = load_file(path)
    enc_sd = {k[len("pulid_encoder."):]: v for k, v in sd.items()
              if k.startswith("pulid_encoder.")}
    ca_sd = {k[len("pulid_ca."):]: v for k, v in sd.items()
             if k.startswith("pulid_ca.")}
    n_ca = 1 + max(int(k.split(".")[0]) for k in ca_sd)
    encoder = IDFormer()
    encoder.load_state_dict(enc_sd)
    cas = torch.nn.ModuleList([CA() for _ in range(n_ca)])
    cas.load_state_dict(ca_sd)
    encoder.eval()
    cas.eval()
    _STATE["adapter"] = (encoder, cas)
    print(f"[flux.pulid_encode] adapter loaded: IDFormer + {n_ca} CA "
          f"({os.path.getsize(path)/1e6:.0f}MB)", flush=True)
    return _STATE["adapter"]


def _load_eva():
    """EVA02-CLIP-L-14-336 视觉塔（CPU 驻留，用时上卡）。"""
    if "eva" in _STATE:
        return _STATE["eva"]
    _ensure_lib()
    from eva_clip import create_model_and_transforms
    from eva_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
    ckpt = os.path.join(PULID_DIR, "EVA02_CLIP_L_336_psz14_s6B.pt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"EVA-CLIP 权重缺失：{ckpt}")
    model, _, _ = create_model_and_transforms(
        "EVA02-CLIP-L-14-336", ckpt, force_custom_clip=True)
    visual = model.visual
    del model  # 文本塔不需要
    mean = getattr(visual, "image_mean", OPENAI_DATASET_MEAN)
    std = getattr(visual, "image_std", OPENAI_DATASET_STD)
    if not isinstance(mean, (list, tuple)):
        mean = (mean,) * 3
    if not isinstance(std, (list, tuple)):
        std = (std,) * 3
    visual.eval()
    _STATE["eva"] = (visual, mean, std)
    print(f"[flux.pulid_encode] EVA02-CLIP-L-14-336 ready "
          f"({os.path.getsize(ckpt)/1e9:.2f}GB)", flush=True)
    return _STATE["eva"]


def _face_helper(device):
    if "helper" not in _STATE:
        _ensure_lib()
        from facexlib.parsing import init_parsing_model
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        wdir = os.path.join(PULID_DIR, "facexlib")
        helper = FaceRestoreHelper(
            upscale_factor=1, face_size=512, crop_ratio=(1, 1),
            det_model="retinaface_resnet50", save_ext="png",
            device=device, model_rootpath=wdir)
        helper.face_parse = None
        helper.face_parse = init_parsing_model(
            model_name="bisenet", device=device, model_rootpath=wdir)
        _STATE["helper"] = helper
        print(f"[flux.pulid_encode] facexlib ready (weights={wdir})",
              flush=True)
    return _STATE["helper"]


def _face_app():
    if "app" not in _STATE:
        from insightface.app import FaceAnalysis
        home = os.environ.get("INSIGHTFACE_HOME", "/models/insightface")
        app = FaceAnalysis(name="antelopev2", root=home,
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _STATE["app"] = app
        print(f"[flux.pulid_encode] insightface antelopev2 ready "
              f"(home={home})", flush=True)
    return _STATE["app"]


def _handler_ante():
    """glintr100 直取特征兜底（insightface 检测不到脸时，在对齐图上取）。"""
    if "handler" not in _STATE:
        import insightface
        home = os.environ.get("INSIGHTFACE_HOME", "/models/insightface")
        p = os.path.join(home, "models", "antelopev2", "glintr100.onnx")
        h = insightface.model_zoo.get_model(
            p, providers=["CPUExecutionProvider"])
        h.prepare(ctx_id=-1)
        _STATE["handler"] = h
    return _STATE["handler"]


# ---------------------------------------------------------------------------
# 图像小工具（移植自 PuLID pulid/utils.py，仅保留需要的两个）
# ---------------------------------------------------------------------------
def _img2tensor(img, bgr2rgb=True):
    import numpy as np
    if bgr2rgb:
        img = img[:, :, [2, 1, 0]]
    return torch.from_numpy(
        np.ascontiguousarray(img.transpose(2, 0, 1))).float()


def _to_gray(img):
    x = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
    return x.repeat(1, 3, 1, 1)


def register(registry):
    @registry.register(
        "flux.pulid_encode",
        inputs={"image": "IMAGE", "face_index": "INT"},
        outputs={"pulid": "FACE", "info": "STRING"},
        description="FLUX PuLID 人脸身份编码（PuLID-FLUX v0.9.1 移植）："
                    "insightface 取 ArcFace 特征 + facexlib 对齐/bisenet 抠脸 "
                    "+ EVA-CLIP 编码 → IDFormer 融合出 id_embedding，连同 20 个 "
                    "PerceiverAttentionCA 打包为 FACE 对象，供 flux.sample 的 "
                    "pulid 输入做身份注入（文本仍主导画面，仅锁定人脸）。"
                    "face_index：-1=最大脸（默认），0..N-1=按面积降序第 N 张")
    def _op_flux_pulid_encode(image, face_index=-1):
        import cv2
        import numpy as np
        from PIL import Image
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import normalize, resize

        if not isinstance(image, Image.Image):
            raise ValueError("flux.pulid_encode: image 需为单张 PIL 图像，"
                             f"got {type(image).__name__}")
        if not torch.cuda.is_available():
            raise ValueError("flux.pulid_encode 需要 CUDA（EVA/IDFormer 上卡）")
        t0 = time.time()
        dev = "cuda"
        wdtype = torch.bfloat16

        # 相位切换：transformer 回 CPU 腾 VRAM
        _move_transformers("cpu")

        np_rgb = np.asarray(image.convert("RGB"))
        image_bgr = cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)

        # 1) insightface antelopev2 身份特征（CPU/onnxruntime）
        app = _face_app()
        face_info = app.get(image_bgr)
        id_ante_embedding = None
        if face_info:
            face_info = sorted(
                face_info,
                key=lambda x: (x["bbox"][2] - x["bbox"][0])
                * (x["bbox"][3] - x["bbox"][1]))
            fi = int(face_index)
            pick = face_info[-1] if fi < 0 else (
                face_info[fi] if fi < len(face_info) else face_info[-1])
            id_ante_embedding = pick["embedding"]
            print(f"[flux.pulid_encode] insightface 检出 {len(face_info)} 张脸，"
                  f"选用 bbox={pick['bbox'].astype(int).tolist()}", flush=True)

        # 2) facexlib 对齐 512x512
        helper = _face_helper(dev)
        helper.clean_all()
        helper.read_image(image_bgr)
        helper.get_face_landmarks_5(only_center_face=True)
        helper.align_warp_face()
        if not helper.cropped_faces:
            raise RuntimeError("facexlib 人脸对齐失败（未检出人脸）")
        align_face = helper.cropped_faces[0]
        if id_ante_embedding is None:
            print("[flux.pulid_encode] insightface 未检出，glintr100 兜底",
                  flush=True)
            id_ante_embedding = _handler_ante().get_feat(align_face)

        id_ante_embedding = torch.from_numpy(id_ante_embedding).to(dev, wdtype)
        if id_ante_embedding.ndim == 1:
            id_ante_embedding = id_ante_embedding.unsqueeze(0)

        # 3) bisenet 解析抠脸（背景置白 + 面部转灰）
        inp = _img2tensor(align_face, bgr2rgb=True).unsqueeze(0) / 255.0
        inp = inp.to(dev)
        with torch.no_grad():
            parsing_out = helper.face_parse(
                normalize(inp, [0.485, 0.456, 0.406],
                          [0.229, 0.224, 0.225]))[0]
        parsing_out = parsing_out.argmax(dim=1, keepdim=True)
        bg_label = [0, 16, 18, 7, 8, 9, 14, 15]
        bg = sum(parsing_out == i for i in bg_label).bool()
        white_image = torch.ones_like(inp)
        face_features_image = torch.where(bg, white_image, _to_gray(inp))

        # 4) EVA-CLIP 编码
        visual, mean, std = _load_eva()
        visual.to(dev, wdtype)
        face_features_image = resize(face_features_image,
                                     visual.image_size,
                                     InterpolationMode.BICUBIC)
        face_features_image = normalize(face_features_image, mean, std)
        with torch.no_grad():
            id_cond_vit, id_vit_hidden = visual(
                face_features_image.to(wdtype),
                return_all_features=False, return_hidden=True, shuffle=False)
        visual.to("cpu")
        id_cond_vit_norm = torch.norm(id_cond_vit, 2, 1, True)
        id_cond_vit = torch.div(id_cond_vit, id_cond_vit_norm)

        # 5) IDFormer 融合
        encoder, cas = _load_adapter()
        encoder.to(dev, wdtype)
        id_cond = torch.cat([id_ante_embedding, id_cond_vit], dim=-1)
        id_vit_hidden = [h.to(dev, wdtype) for h in id_vit_hidden]
        with torch.no_grad():
            id_embedding = encoder(id_cond, id_vit_hidden)
        encoder.to("cpu")

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        info = (f"{time.time()-t0:.1f}s id_embedding={tuple(id_embedding.shape)} "
                f"ca={len(cas)} face={'insightface' if face_info else 'glintr100'}")
        print(f"[flux.pulid_encode] {info}", flush=True)
        return {"pulid": {"id_embedding": id_embedding.float().cpu(),
                          "ca": cas},
                "info": info}
