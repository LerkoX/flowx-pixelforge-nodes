"""ControlNet G7 预处理器族（anyline / lineart / softedge / scribble / mlsd / normal / shuffle）。

为什么单独一个族文件：服务端插件上传以 `op_name.replace(".", "_") + ".py"` 命名，
`preprocess.canny/depth/openpose` 三算子已由 `preprocess_canny.py` 拥有——同族文件里
再注册它们会撞 409（跨文件重名，全量回滚）。故本族只注册 G7 的 7 个新算子，
与三算子族文件 `preprocess.py` 并存；同族多算子共处一个文件，节点侧只需一次上传。

G7 族统一口径：输入 IMAGE → 输出同尺寸 RGB 提示图（黑底白线/彩图惯例），
检测器一律 CPU 推理 + 进程级单例缓存（8GB 卡显存留给 UNet/VAE），
权重从 PREPROCESSOR_MODELS_DIR（默认 /models/preprocessors/<name>）本地加载，
零网络访问；缺权重时报错直接点名缺哪个文件（清单见 nodes/README.md）。

依赖：controlnet_aux 0.0.10（已烘焙进 service 镜像）；一律函数内懒加载。
分发：本文件是节点共享算子族（单一事实源），由 nodes/_tools/sync-common.py
分发为 7 个节点各自的 server_op.py，节点运行时经 ensure_plugin 自注册（hash 幂等）。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "42636b08e915"
OP_FAMILY = "preprocess_g7"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

"""ControlNet 预处理器算子（第三档批次 2/3 + G7 补齐，插件化交付）。

- preprocess.canny：cv2.Canny 边缘线稿（黑底白线，ControlNet canny 出图惯例）
- preprocess.openpose：OpenPose 骨架图（controlnet_aux，body 常驻，hand/face 可选）
- preprocess.depth：MiDaS dpt_hybrid 深度图（批次 3；control_v11f1p_sd15_depth
  训练时用的就是 MiDaS dpt_hybrid，预处理器与权重最对口）
- preprocess.anyline：Anyline（MTEED）高精度线稿（配 MistoLine / anyline ControlNet）
- preprocess.lineart：线稿三档（variant=lineart|anime|standard）
- preprocess.softedge：软边缘（variant=pidi|hed，配 control_v11p_sd15_softedge）
- preprocess.scribble：涂鸦稿（PidiNet safe+scribble，配 control_v11p_sd15_scribble）
- preprocess.mlsd：M-LSD 直线检测（建筑/室内，配 control_v11p_sd15_mlsd）
- preprocess.normal：NormalBae 法线图（配 control_v11p_sd15_normalbae）
- preprocess.shuffle：内容打乱（ContentShuffle，无权重；配 control_v11e_sd15_shuffle）

G7 族统一口径：输入 IMAGE → 输出同尺寸 RGB 提示图（黑底白线/彩线惯例），
检测器一律 CPU 推理、进程级单例缓存（8GB 卡显存留给 UNet/VAE），
权重从 PREPROCESSOR_MODELS_DIR（默认 /models/preprocessors/<name>）本地加载，
零网络访问；缺权重时报错直接给出缺哪个文件（见 nodes/README.md 权重清单）。

用途：闭环"人物替换保动作"——load-image(真人照片) → preprocess.openpose
→ sd.controlnet.apply(openpose 权重) → sd.sample(换服装/风格提示词)，姿势由骨架锁定。
depth 则锁构图/空间层次（室内、产品、场景改造）。

分发：本文件是节点共享算子族（单一事实源），由 nodes/_tools/sync-common.py
分发为各成员节点的 server_op.py，节点运行时经 ensure_plugin 自注册到推理服务
（POST /admin/plugins，hash 幂等）。使用：对应专属节点或 inference-op 通用节点。

依赖：opencv-python-headless（canny）、controlnet_aux（openpose），均烘焙进镜像；
numpy/cv2/controlnet_aux 一律函数内懒加载（本地无 GPU 契约测试可导入）。
权重：openpose 预处理器从 PREPROCESSOR_MODELS_DIR（默认 /models/preprocessors）
本地加载，不访问网络（宿主机直下 body_pose_model.pth / hand_pose_model.pth）。
"""
import os

from PIL import Image

PREPROC_DIR = os.environ.get("PREPROCESSOR_MODELS_DIR", "/models/preprocessors")


def _as_image(image, op_name):
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"{op_name}: image 需为 IMAGE 对象，got {type(image).__name__}")
    return image


# --------------------------------------------------------------- G7 预处理器族
# 检测器进程级单例：{spec_key: detector}。除 canny/cv2 外全部 CPU 推理。
_DETECTORS = {}

# spec_key -> (控制的类名, 权重子目录, 必需文件名列表)
# 类名 None ⇒ 无权重构造（ControlNet 的 standard/shuffle 是纯算法）
_DETECTOR_SPECS = {
    "anyline":         ("AnylineDetector",      "anyline",       ["MTEED.pth"]),
    "lineart":         ("LineartDetector",      "lineart",       ["sk_model.pth", "sk_model2.pth"]),
    "lineart_anime":   ("LineartAnimeDetector", "lineart_anime", ["netG.pth"]),
    "lineart_standard": ("LineartStandardDetector", None,        []),
    "pidi":            ("PidiNetDetector",      "pidi",          ["table5_pidinet.pth"]),
    "hed":             ("HEDdetector",          "hed",           ["ControlNetHED.pth"]),
    "mlsd":            ("MLSDdetector",         "mlsd",          ["mlsd_large_512_fp32.pth"]),
    "normalbae":       ("NormalBaeDetector",    "normalbae",     ["scannet.pt"]),
    "shuffle":         ("ContentShuffleDetector", None,          []),
}

# anyline 的权重文件名与类默认值不同（仓库里叫 Anyline/MTEED.pth），需显式传
_EXTRA_FILENAME = {"anyline": "MTEED.pth"}


def _detector(spec_key, op_name):
    """按 spec 取检测器（进程级单例）。权重缺失/目录缺失时报可操作的错。"""
    if spec_key in _DETECTORS:
        return _DETECTORS[spec_key]
    cls_name, subdir, files = _DETECTOR_SPECS[spec_key]
    import controlnet_aux  # 懒加载
    cls = getattr(controlnet_aux, cls_name)
    kwargs = {}
    if subdir:
        path = os.path.join(PREPROC_DIR, subdir)
        missing = [f for f in files
                   if not os.path.isfile(os.path.join(path, f))]
        if missing:
            raise FileNotFoundError(
                f"{op_name}: 预处理器权重缺失：{path}/{'、'.join(missing)}"
                f"（把 {'、'.join(missing)} 放到 {path}/ 后重试；"
                f"权重清单见 nodes/README.md）")
        # 目录模式：from_pretrained 直接 join(dir, filename)，零网络访问；
        # local_files_only 仅在类签名支持时透传（有的类只收 filename/subfolder）
        import inspect
        params = inspect.signature(cls.from_pretrained).parameters
        if "local_files_only" in params:
            kwargs["local_files_only"] = True
        if spec_key in _EXTRA_FILENAME:
            kwargs["filename"] = _EXTRA_FILENAME[spec_key]
        det = cls.from_pretrained(path, **kwargs)
        print(f"[{op_name}] 检测器已加载：{path}", flush=True)
    else:
        det = cls()
        print(f"[{op_name}] 检测器已构造（无权重）", flush=True)
    _DETECTORS[spec_key] = det
    return det


def _finish(out, img, op_name, extra=""):
    """检测器输出统一收口：→ RGB，尺寸回缩到输入尺寸（下游 controlnet.apply
    的 hint 必须与出图尺寸对齐），打印一行统计便于日志排障。"""
    if not isinstance(out, Image.Image):
        import numpy as np
        out = Image.fromarray(np.asarray(out))
    out = out.convert("RGB")
    if out.size != img.size:
        out = out.resize(img.size, Image.BILINEAR)
    print(f"[{op_name}] {img.size} -> {out.size} {extra}", flush=True)
    return {"image": out}


def preprocess_anyline(image, detect_resolution=1280, guassian_sigma=2.0,
                       intensity_threshold=3):
    """Anyline（MTEED）高精度线稿：真人照片/插画 → 黑底白线细致线稿。
    比 lineart/canny 保留更多纹理细节，配 MistoLine/anyline 系 ControlNet。
    detect_resolution 默认 1280（Anyline 官方口径，越大越细但越慢）；
    guassian_sigma/intensity_threshold 为后处理去噪阈值（参数名拼写沿用上游）。"""
    op = "preprocess.anyline"
    img = _as_image(image, op)
    det = _detector("anyline", op)
    out = det(img.convert("RGB"),
              detect_resolution=max(64, int(detect_resolution)),
              guassian_sigma=float(guassian_sigma),
              intensity_threshold=int(intensity_threshold))
    return _finish(out, img, op,
                   f"sigma={float(guassian_sigma)} thr={int(intensity_threshold)}")


def preprocess_lineart(image, variant="lineart", detect_resolution=512,
                       coarse=False):
    """线稿提取（三档）：variant=lineart（写实插画线稿，coarse=1 更粗松）
    / anime（动漫线稿，netG）/ standard（无需权重，最轻量）。
    配 control_v11p_sd15_lineart / lineart_anime。"""
    op = "preprocess.lineart"
    img = _as_image(image, op)
    v = str(variant).strip().lower()
    res = max(64, int(detect_resolution))
    if v in ("lineart", "sk", "sketch", "realistic"):
        det = _detector("lineart", op)
        out = det(img.convert("RGB"), coarse=bool(coarse),
                  detect_resolution=res, image_resolution=res)
        extra = f"variant=lineart coarse={bool(coarse)}"
    elif v in ("anime", "lineart_anime"):
        det = _detector("lineart_anime", op)
        out = det(img.convert("RGB"), detect_resolution=res,
                  image_resolution=res)
        extra = "variant=anime"
    elif v in ("standard", "lineart_standard"):
        det = _detector("lineart_standard", op)
        # 该类上游只收 detect_resolution（无 image_resolution 形参）
        out = det(img.convert("RGB"), detect_resolution=res)
        extra = "variant=standard"
    else:
        raise ValueError(
            f"{op}: variant={variant!r} 不支持（可用：lineart / anime / standard）")
    return _finish(out, img, op, extra)


def preprocess_softedge(image, variant="pidi", detect_resolution=512,
                        safe=True):
    """软边缘/边缘提取：variant=pidi（PidiNet，safe=1 出平滑软边，默认）
    / hed（HED，纹理更足）。配 control_v11p_sd15_softedge。"""
    op = "preprocess.softedge"
    img = _as_image(image, op)
    v = str(variant).strip().lower()
    res = max(64, int(detect_resolution))
    if v in ("pidi", "pidinet"):
        det = _detector("pidi", op)
        out = det(img.convert("RGB"), detect_resolution=res,
                  image_resolution=res, safe=bool(safe))
        extra = f"variant=pidi safe={bool(safe)}"
    elif v == "hed":
        det = _detector("hed", op)
        out = det(img.convert("RGB"), detect_resolution=res,
                  image_resolution=res, safe=bool(safe))
        extra = f"variant=hed safe={bool(safe)}"
    else:
        raise ValueError(f"{op}: variant={variant!r} 不支持（可用：pidi / hed）")
    return _finish(out, img, op, extra)


def preprocess_scribble(image, detect_resolution=512, safe=True):
    """涂鸦稿（scribble）：PidiNet 的 scribble 模式（safe=1 走 safe_step 平滑），
    产出黑底白涂鸦线，配 control_v11p_sd15_scribble 做手绘控制。"""
    op = "preprocess.scribble"
    img = _as_image(image, op)
    det = _detector("pidi", op)
    res = max(64, int(detect_resolution))
    out = det(img.convert("RGB"), detect_resolution=res,
              image_resolution=res, safe=bool(safe), scribble=True)
    return _finish(out, img, op, f"safe={bool(safe)} scribble=True")


def preprocess_mlsd(image, detect_resolution=512, thr_v=0.1, thr_d=0.1):
    """M-LSD 直线检测：建筑/室内/工业设计 → 黑底白直线图。
    thr_v/thr_d 为竖/横线置信阈值（调高线更少更干净），
    配 control_v11p_sd15_mlsd。"""
    op = "preprocess.mlsd"
    img = _as_image(image, op)
    det = _detector("mlsd", op)
    res = max(64, int(detect_resolution))
    out = det(img.convert("RGB"), detect_resolution=res, image_resolution=res,
              thr_v=float(thr_v), thr_d=float(thr_d))
    return _finish(out, img, op, f"thr_v={float(thr_v)} thr_d={float(thr_d)}")


def preprocess_normal(image, detect_resolution=512):
    """NormalBae 法线图：RGB → 法线贴图风格（蓝紫基底），锁光照/立体结构，
    配 control_v11p_sd15_normalbae；权重 scannet.pt。"""
    op = "preprocess.normal"
    img = _as_image(image, op)
    det = _detector("normalbae", op)
    res = max(64, int(detect_resolution))
    out = det(img.convert("RGB"), detect_resolution=res,
              image_resolution=res)
    return _finish(out, img, op)


def preprocess_shuffle(image, detect_resolution=512):
    """ContentShuffle 内容打乱：把画面切成小块随机重排（无权重，纯算法），
    配 control_v11e_sd15_shuffle 做"保构图/材质、换内容"的迁移。"""
    op = "preprocess.shuffle"
    img = _as_image(image, op)
    det = _detector("shuffle", op)
    res = max(64, int(detect_resolution))
    out = det(img.convert("RGB"), detect_resolution=res,
              image_resolution=res)
    return _finish(out, img, op)


def register(registry):
    """向算子注册表登记 G7 预处理器族（7 个算子；单文件承载，节点侧一次上传）。"""
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "preprocess.anyline",
        inputs={"image": "IMAGE", "detect_resolution": "INT",
                "guassian_sigma": "FLOAT", "intensity_threshold": "INT"},
        outputs={"image": "IMAGE"},
        description="Anyline（MTEED）高精度线稿：IMAGE → 黑底白线细致线稿，"
                    "配 MistoLine/anyline ControlNet；detect_resolution 默认 1280，"
                    "权重 /models/preprocessors/anyline/MTEED.pth")(preprocess_anyline)
    registry.register(
        "preprocess.lineart",
        inputs={"image": "IMAGE", "variant": "STRING",
                "detect_resolution": "INT", "coarse": "BOOL"},
        outputs={"image": "IMAGE"},
        description="线稿提取：variant=lineart（写实，coarse 控粗细）/anime"
                    "（动漫）/standard（无权重轻量）；配 lineart ControlNet")(preprocess_lineart)
    registry.register(
        "preprocess.softedge",
        inputs={"image": "IMAGE", "variant": "STRING",
                "detect_resolution": "INT", "safe": "BOOL"},
        outputs={"image": "IMAGE"},
        description="软边缘提取：variant=pidi（PidiNet 平滑软边，默认）/hed"
                    "（纹理更足）；配 control_v11p_sd15_softedge")(preprocess_softedge)
    registry.register(
        "preprocess.scribble",
        inputs={"image": "IMAGE", "detect_resolution": "INT",
                "safe": "BOOL"},
        outputs={"image": "IMAGE"},
        description="涂鸦稿：PidiNet scribble 模式 → 黑底白涂鸦线，"
                    "配 control_v11p_sd15_scribble")(preprocess_scribble)
    registry.register(
        "preprocess.mlsd",
        inputs={"image": "IMAGE", "detect_resolution": "INT",
                "thr_v": "FLOAT", "thr_d": "FLOAT"},
        outputs={"image": "IMAGE"},
        description="M-LSD 直线检测：建筑/室内 → 黑底白直线（thr_v/thr_d "
                    "控竖/横线阈值），配 control_v11p_sd15_mlsd")(preprocess_mlsd)
    registry.register(
        "preprocess.normal",
        inputs={"image": "IMAGE", "detect_resolution": "INT"},
        outputs={"image": "IMAGE"},
        description="NormalBae 法线图：RGB → 法线贴图（锁光照/立体结构），"
                    "配 control_v11p_sd15_normalbae；权重 scannet.pt")(preprocess_normal)
    registry.register(
        "preprocess.shuffle",
        inputs={"image": "IMAGE", "detect_resolution": "INT"},
        outputs={"image": "IMAGE"},
        description="ContentShuffle 内容打乱（无权重纯算法）：保构图换内容，"
                    "配 control_v11e_sd15_shuffle")(preprocess_shuffle)
