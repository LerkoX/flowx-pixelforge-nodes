"""人像美化算子族（beautify.auto，朋友圈九宫格 v2）。

- beautify.auto：IMAGE → 智能裁方 + 皮肤美白磨皮 + 大眼 + 瘦脸 + meta JSON

实现要点：
- 人脸检测/关键点：insightface antelopev2（onnxruntime CPU，INSIGHTFACE_HOME，
  与 instantid 族同模型包）；用 **landmark_3d_68**（68 点布局是 iBUG/dlib 通用约定：
  下颌 0-16、左眼 36-41、右眼 42-47、鼻 27-35、嘴 48-67），比 106 点的索引口径稳定；
- 美白磨皮：YCrCb 椭圆肤色模型 → 形态学开运算 + 高斯软掩码 → 双边滤波按掩码混合
  磨皮 + LAB L 通道按掩码提亮（只动皮肤，衣物/背景不动）；
- 大眼：以 6 点眼廓中心为圆心、眼宽 1.1 倍为半径做径向放大（remap，smoothstep
  衰减，圆外像素不动）；
- 瘦脸：下颌两侧点（2-7、9-14）向鼻尖方向内推（幅度随 slim × 脸宽），眉/眼/鼻/
  嘴/下巴尖作固定锚，Shepard IDW 在 1/4 下采样网格上插值位移场再上采样 remap
  （场天然平滑，无需 TPS）；
- 智能裁方：以人脸 bbox 中心为锚 cover 裁正方形（脸放在裁剪框纵向 42% 处）；
  无脸回退居中裁；crop=0 不裁。

分发：本文件是节点共享算子族（单一事实源），由 nodes/_tools/sync-common.py
分发为成员节点的 server_op.py，节点运行时经 ensure_plugin 自注册。
依赖：cv2(headless) / numpy / insightface——一律函数内懒加载。
"""

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "1ff46762b0e4"
OP_FAMILY = "beautify"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

import json
import os

from PIL import Image

_FACE_APP = None


def _dumps(obj):
    """JSON 负载的信封编码：`@` + JSON（防 Studio 自动反序列化，见 seg.py）。"""
    return "@" + json.dumps(obj, ensure_ascii=False)


def _face_app():
    """FaceAnalysis 惰性单例（onnxruntime CPU；与 instantid 族同一 antelopev2 包，
    但实例独立——插件间不共享进程内单例）。"""
    global _FACE_APP
    if _FACE_APP is None:
        from insightface.app import FaceAnalysis  # 懒加载：契约测试可导入
        home = os.environ.get("INSIGHTFACE_HOME", "/models/insightface")
        app = FaceAnalysis(name="antelopev2", root=home,
                           providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))
        _FACE_APP = app
        print(f"[beautify] insightface antelopev2 ready (home={home})", flush=True)
    return _FACE_APP


def _detect(bgr, det_thresh):
    """最大脸检测 → (bbox, lmk68)；无脸 (None, None)，无关键点 (bbox, None)。"""
    import cv2
    import numpy as np
    app = _face_app()
    app.det_model.det_thresh = float(det_thresh)
    faces = app.get(bgr)
    if not faces:
        return None, None
    f = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
    bbox = [float(v) for v in f.bbox]
    lmk = getattr(f, "landmark_3d_68", None)
    pts = None
    if lmk is not None:
        pts = np.asarray(lmk, dtype=np.float32)[:, :2]  # (68,2)，丢 z 深度
    return bbox, pts


def _skin_mask(bgr):
    """YCrCb 椭圆肤色模型 → 0~1 软掩码（float32）。"""
    import cv2
    import numpy as np
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    _, cr, cb = cv2.split(ycrcb)
    mask = ((cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.GaussianBlur(mask, (0, 0), 5)
    return mask.astype(np.float32) / 255.0


def _whiten_smooth(bgr, whiten, smooth, mask):
    """磨皮（双边滤波按肤掩码混合）+ 提亮（LAB L 通道按肤掩码加性）。"""
    import cv2
    import numpy as np
    out = bgr.astype(np.float32)
    m = mask[..., None]
    if smooth > 0:
        sm = cv2.bilateralFilter(bgr, 9, 40, 40).astype(np.float32)
        out = out * (1 - m * smooth) + sm * (m * smooth)
    if whiten > 0:
        lab = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
        L = lab[..., 0].astype(np.float32)
        L = np.clip(L + whiten * 28.0 * mask, 0, 255)
        lab[..., 0] = L.astype(np.uint8)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _radial_zoom(bgr, center, radius, amount):
    """以 center 为心、radius 内径向放大（大眼）；amount 0~0.4，圆外不动。"""
    import cv2
    import numpy as np
    cx, cy = float(center[0]), float(center[1])
    h0, w0 = bgr.shape[:2]
    x0, y0 = max(0, int(cx - radius - 2)), max(0, int(cy - radius - 2))
    x1, y1 = min(w0, int(cx + radius + 3)), min(h0, int(cy + radius + 3))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return bgr
    sub = bgr[y0:y1, x0:x1]
    h, w = sub.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    dx, dy = gx - (cx - x0), gy - (cy - y0)
    d = np.sqrt(dx * dx + dy * dy)
    t = np.clip(1.0 - d / float(radius), 0.0, 1.0)
    k = 1.0 - amount * (t * t * (3 - 2 * t))  # smoothstep 衰减，中心放大最多
    mapx = ((cx - x0) + dx * k).astype(np.float32)
    mapy = ((cy - y0) + dy * k).astype(np.float32)
    warped = cv2.remap(sub, mapx, mapy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    m = (t > 0).astype(np.float32)[..., None]
    bgr[y0:y1, x0:x1] = sub * (1 - m) + warped * m
    return bgr


def _slim_face(bgr, pts, slim):
    """下颌内推（Shepard IDW 位移场，1/4 下采样计算再上采样 remap）。"""
    import cv2
    import numpy as np
    nose = np.asarray(pts[30], dtype=np.float32)
    face_w = float(np.linalg.norm(pts[0] - pts[16]))
    amp = float(slim) * face_w * 0.25
    if amp < 0.5:
        return bgr
    src, dst = [], []
    for i in list(range(2, 8)) + list(range(9, 15)):  # 下颌两侧（下巴尖 8 不动）
        p = np.asarray(pts[i], dtype=np.float32)
        inward = nose - p
        n = float(np.linalg.norm(inward))
        if n < 1:
            continue
        inward /= n
        w = 1.0 - abs(i - 8) / 8.0  # 越靠下巴幅度越小
        src.append(p)
        dst.append(p + inward * (amp * w))
    # 固定锚（位移=0）：耳角/下巴尖/眉/眼/鼻/嘴——约束场不外溢
    for i in (0, 16, 8, 19, 24, 27, 30, 33, 36, 45, 39, 42, 48, 54, 51, 57, 62, 66):
        p = np.asarray(pts[i], dtype=np.float32)
        src.append(p)
        dst.append(p.copy())
    src = np.stack(src)
    disp = np.stack(dst) - src
    h, w = bgr.shape[:2]
    gs = 4
    gw, gh = w // gs + 1, h // gs + 1
    gx, gy = np.meshgrid(np.arange(gw) * gs, np.arange(gh) * gs)
    grid = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)  # (N,2)
    d2 = ((grid[:, None, :] - src[None, :, :]) ** 2).sum(-1)  # (N,A)
    wgt = 1.0 / np.maximum(d2, 1.0)  # 1/d² Shepard，远处自然衰减
    field = (wgt @ disp) / np.maximum(wgt.sum(1, keepdims=True), 1e-9)
    fx = field[:, 0].reshape(gh, gw).astype(np.float32)
    fy = field[:, 1].reshape(gh, gw).astype(np.float32)
    fx = cv2.resize(fx, (w, h), interpolation=cv2.INTER_LINEAR)
    fy = cv2.resize(fy, (w, h), interpolation=cv2.INTER_LINEAR)
    gx_f, gy_f = np.meshgrid(np.arange(w, dtype=np.float32),
                             np.arange(h, dtype=np.float32))
    # remap 语义：out(x,y) = in(mapx,mapy)；内容位移 = field ⇒ 反向采样
    return cv2.remap(bgr, (gx_f - fx).astype(np.float32),
                     (gy_f - fy).astype(np.float32), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def _square_crop(bgr, bbox, out_size):
    """以人脸 bbox 中心为锚 cover 裁正方形（脸在裁剪框纵向 42%）；无脸居中。"""
    import cv2
    h, w = bgr.shape[:2]
    side = min(w, h)
    if bbox:
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        x0 = int(round(min(max(cx - side / 2.0, 0), w - side)))
        y0 = int(round(min(max(cy - side * 0.42, 0), h - side)))
    else:
        x0, y0 = (w - side) // 2, (h - side) // 2
    crop = bgr[y0:y0 + side, x0:x0 + side]
    if out_size > 0 and out_size != side:
        crop = cv2.resize(crop, (out_size, out_size),
                          interpolation=cv2.INTER_LANCZOS4)
    return crop


def beautify_auto(image, crop=800, whiten=0.5, smooth=0.4, eye=0.12,
                  slim=0.10, det_thresh=0.3):
    """单张人像美化：裁方 + 美白磨皮 + 大眼 + 瘦脸（无脸/无关键点自动降级）。"""
    if not isinstance(image, Image.Image):
        raise ValueError(f"beautify.auto: image 需为单张 PIL 图像，got {type(image).__name__}")
    import cv2
    import numpy as np
    rgb = image.convert("RGB")
    out = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)
    bbox, pts = _detect(out, float(det_thresh))
    meta = {"face": bool(bbox), "landmarks": pts is not None}
    whiten, smooth = float(whiten), float(smooth)
    if whiten > 0 or smooth > 0:
        out = _whiten_smooth(out, whiten, smooth, _skin_mask(out))
    if pts is not None:
        eye = float(eye)
        if eye > 0:
            for a, b in ((36, 41), (42, 47)):  # iBUG 68：左/右眼廓 6 点
                ep = pts[a:b + 1]
                center = ep.mean(axis=0)
                eye_w = float(np.linalg.norm(pts[a] - pts[a + 3]))
                out = _radial_zoom(out, center, max(10.0, eye_w * 1.1), eye)
        slim = float(slim)
        if slim > 0:
            out = _slim_face(out, pts, slim)
    crop = int(crop)
    if crop > 0:
        out = _square_crop(out, bbox, crop)
        meta["cropped"] = True
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    print(f"[beautify.auto] {image.width}x{image.height} -> {pil.width}x{pil.height} "
          f"face={meta['face']} lmk={meta['landmarks']}", flush=True)
    return {"image": pil, "width": pil.width, "height": pil.height,
            "meta": _dumps(meta)}


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "beautify.auto",
        inputs={"image": "IMAGE", "crop": "INT", "whiten": "FLOAT",
                "smooth": "FLOAT", "eye": "FLOAT", "slim": "FLOAT",
                "det_thresh": "FLOAT"},
        outputs={"image": "IMAGE", "width": "INT", "height": "INT",
                 "meta": "STRING"},
        description="人像美化（九宫格 v2）：智能裁方（人脸中心 cover，0=不裁）"
                    "+ 皮肤美白磨皮（YCrCb 肤掩码 + 双边滤波 + LAB 提亮）"
                    "+ 大眼（68 点眼廓径向放大）+ 瘦脸（下颌 IDW 内推）。"
                    "insightface antelopev2（INSIGHTFACE_HOME，CPU）"
    )(beautify_auto)
