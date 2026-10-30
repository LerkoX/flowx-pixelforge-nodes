"""ControlNet 预处理器算子（第三档批次 2/3，插件化交付）。

- preprocess.canny：cv2.Canny 边缘线稿（黑底白线，ControlNet canny 出图惯例）
- preprocess.openpose：OpenPose 骨架图（controlnet_aux，body 常驻，hand/face 可选）
- preprocess.depth：MiDaS dpt_hybrid 深度图（批次 3；control_v11f1p_sd15_depth
  训练时用的就是 MiDaS dpt_hybrid，预处理器与权重最对口）

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

# --- F1 族版本化（dev-plan §47）：算子身份 = 实现内容 ------------------------
# 注册名统一改写为 "<基名>@<OP_VERSION>"（族内 register() 调用不用改），
# OP_VERSION 由 nodes/_tools/sync-common.py 依本文件内容自动改写（幂等），禁止手改。
# 改过实现的族 ⇒ 换名换文件 ⇒ 新旧版本在服务端并存、互不顶替（旧版顶不掉新版）。
OP_VERSION = "11db1c7a57a0"
OP_FAMILY = "preprocess"  # 必须等于本文件 stem（gate J 校验）；落盘名 <OP_FAMILY>_<OP_VERSION>.py


class _VersionedRegistry:
    """包装注册表：族内注册名一律带上版本后缀。"""

    def __init__(self, inner, version):
        self._inner = inner
        self._version = version

    def register(self, name, inputs, outputs, description=""):
        return self._inner.register(f"{name}@{self._version}", inputs, outputs,
                                    description)
# ---------------------------------------------------------------------------

import os

from PIL import Image

PREPROC_DIR = os.environ.get("PREPROCESSOR_MODELS_DIR", "/models/preprocessors")


def _as_image(image, op_name):
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"{op_name}: image 需为 IMAGE 对象，got {type(image).__name__}")
    return image


def preprocess_canny(image, low_threshold=100, high_threshold=200):
    """Canny 边缘检测：IMAGE → 黑底白线线稿 IMAGE（ControlNet canny 提示图惯例）。
    low/high_threshold 为 cv2.Canny 双阈值（0-255，low < high）；线条太密调高 low，
    断线太多调低 low。"""
    img = _as_image(image, "preprocess.canny")
    low, high = int(low_threshold), int(high_threshold)
    if not (0 <= low < high <= 255):
        raise ValueError(
            f"preprocess.canny: 需 0 <= low < high <= 255，got ({low},{high})")
    import cv2  # 懒加载
    import numpy as np
    arr = np.asarray(img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)  # 单通道 0/255
    out = Image.fromarray(edges).convert("RGB")
    ratio = float((np.asarray(edges) > 0).mean())
    print(f"[preprocess.canny] {img.size} thr=({low},{high}) "
          f"edge_ratio={ratio:.3f}", flush=True)
    return {"image": out}


_openpose_detector = None


def _get_openpose():
    """openpose 检测器进程级单例（权重常驻内存 ~360MB，避免每次调用重载）。"""
    global _openpose_detector
    if _openpose_detector is None:
        from controlnet_aux import OpenposeDetector  # 懒加载
        path = os.path.join(PREPROC_DIR, "openpose")
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"preprocess.openpose: 预处理器权重目录不存在：{path}"
                f"（需 body_pose_model.pth / hand_pose_model.pth）")
        _openpose_detector = OpenposeDetector.from_pretrained(
            path, local_files_only=True)
        print(f"[preprocess.openpose] 检测器已加载：{path}", flush=True)
    return _openpose_detector


# ===================== DWPose（ONNX 后端） =====================
# 背景：镜像内 controlnet_aux==0.0.10 的 DWposeDetector 走 mmpose/mmcv/mmdet
# 栈（镜像未安装），故内联 Fannovel16/comfyui_controlnet_aux 的 ONNX 推理
# （yolox_l 人检 + dw-ll_ucoco_384 关键点，onnxruntime CPU 即可），绘图则
# 复用 controlnet_aux 的 DWposeDetector.__call__（构造空壳实例、把
# pose_estimation 换成本 ONNX Wholebody，输出 (candidate, subset) 与
# mmpose 版同构）。对侧身回头/遮挡/交叉肢体显著优于标准 OpenPose。
# 权重：/models/preprocessors/dwpose/{yolox_l.onnx, dw-ll_ucoco_384.onnx}
# 移植源（MIT）：github.com/Fannovel16/comfyui_controlnet_aux
#   src/custom_controlnet_aux/dwpose/{dw_onnx/cv_ox_det.py, dw_onnx/cv_ox_pose.py,
#   wholebody.py}（裁剪为 onnxruntime-only，懒加载依赖）


def _dw_deps():
    import cv2  # 懒加载（本地无 cv2 契约测试需可导入本文件）
    import numpy as np
    return cv2, np


# --- vendored: cv_ox_det.py（yolox onnx 人检，640 letterbox + 多类 NMS） ---

def _dw_nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    _, np = _dw_deps()
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]
    return keep


def _dw_multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    _, np = _dw_deps()
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        valid_scores = cls_scores[valid_score_mask]
        valid_boxes = boxes[valid_score_mask]
        keep = _dw_nms(valid_boxes, valid_scores, nms_thr)
        if len(keep) > 0:
            cls_inds = np.ones((len(keep), 1)) * cls_ind
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1)
            final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)


def _dw_demo_postprocess(outputs, img_size, p6=False):
    _, np = _dw_deps()
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))
    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def _dw_det_preprocess(img, input_size, swap=(2, 0, 1)):
    cv2, np = _dw_deps()
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3),
                             dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img, (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def _dw_inference_detector(session, oriImg, detect_classes=(0,),
                           dtype=None):
    """yolox_l.onnx 人检：BGR 图 → (N,4) xyxy 框（默认只留 person 类）。"""
    _, np = _dw_deps()
    if dtype is None:
        dtype = np.float32
    input_shape = (640, 640)
    img, ratio = _dw_det_preprocess(oriImg, input_shape)
    inp = img[None, :, :, :].astype(dtype)
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: inp})
    predictions = _dw_demo_postprocess(output[0], input_shape)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]
    boxes_xyxy = np.ones_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
    boxes_xyxy /= ratio
    dets = _dw_multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
    if dets is None:
        return None
    final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
    isscore = final_scores > 0.3
    iscat = np.isin(final_cls_inds, list(detect_classes))
    isbbox = [i and j for (i, j) in zip(isscore, iscat)]
    final_boxes = final_boxes[isbbox]
    return final_boxes


# --- vendored: cv_ox_pose.py（dwpose onnx 133 关键点，simcc 解码） ---

def _dw_bbox_xyxy2cs(bbox, padding=1.):
    _, np = _dw_deps()
    dim = bbox.ndim
    if dim == 1:
        bbox = bbox[None, :]
    x1, y1, x2, y2 = np.hsplit(bbox, [1, 2, 3])
    center = np.hstack([x1 + x2, y1 + y2]) * 0.5
    scale = np.hstack([x2 - x1, y2 - y1]) * padding
    if dim == 1:
        center = center[0]
        scale = scale[0]
    return center, scale


def _dw_fix_aspect_ratio(bbox_scale, aspect_ratio):
    _, np = _dw_deps()
    w, h = np.hsplit(bbox_scale, [1])
    bbox_scale = np.where(w > h * aspect_ratio,
                          np.hstack([w, w / aspect_ratio]),
                          np.hstack([h * aspect_ratio, h]))
    return bbox_scale


def _dw_rotate_point(pt, angle_rad):
    _, np = _dw_deps()
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    rot_mat = np.array([[cs, -sn], [sn, cs]])
    return rot_mat @ pt


def _dw_get_3rd_point(a, b):
    _, np = _dw_deps()
    direction = a - b
    c = b + np.r_[-direction[1], direction[0]]
    return c


def _dw_get_warp_matrix(center, scale, rot, output_size, shift=(0., 0.),
                        inv=False):
    cv2, np = _dw_deps()
    shift = np.array(shift)
    src_w = scale[0]
    dst_w = output_size[0]
    dst_h = output_size[1]
    rot_rad = np.deg2rad(rot)
    src_dir = _dw_rotate_point(np.array([0., src_w * -0.5]), rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    src[2, :] = _dw_get_3rd_point(src[0, :], src[1, :])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _dw_get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        warp_mat = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        warp_mat = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return warp_mat


def _dw_top_down_affine(input_size, bbox_scale, bbox_center, img):
    cv2, np = _dw_deps()
    w, h = input_size
    warp_size = (int(w), int(h))
    bbox_scale = _dw_fix_aspect_ratio(bbox_scale, aspect_ratio=w / h)
    warp_mat = _dw_get_warp_matrix(bbox_center, bbox_scale, 0,
                                   output_size=(w, h))
    img = cv2.warpAffine(img, warp_mat, warp_size, flags=cv2.INTER_LINEAR)
    return img, bbox_scale


def _dw_pose_preprocess(img, out_bbox, input_size=(192, 256)):
    """按人框逐个仿射裁剪 + ImageNet 归一化。返回 (imgs, centers, scales)。"""
    _, np = _dw_deps()
    img_shape = img.shape[:2]
    out_img, out_center, out_scale = [], [], []
    if len(out_bbox) == 0:
        out_bbox = [[0, 0, img_shape[1], img_shape[0]]]
    for i in range(len(out_bbox)):
        x0, y0, x1, y1 = out_bbox[i][0], out_bbox[i][1], \
            out_bbox[i][2], out_bbox[i][3]
        bbox = np.array([x0, y0, x1, y1])
        center, scale = _dw_bbox_xyxy2cs(bbox, padding=1.25)
        resized_img, scale = _dw_top_down_affine(input_size, scale, center, img)
        mean = np.array([123.675, 116.28, 103.53])
        std = np.array([58.395, 57.12, 57.375])
        resized_img = (resized_img - mean) / std
        out_img.append(resized_img)
        out_center.append(center)
        out_scale.append(scale)
    return out_img, out_center, out_scale


def _dw_pose_inference(sess, img, dtype=None):
    """批量跑 dwpose onnx（onnxruntime 支持 batch）。"""
    _, np = _dw_deps()
    if dtype is None:
        dtype = np.float32
    inp = np.stack(img, axis=0).transpose(0, 3, 1, 2).astype(dtype)
    input_name = sess.get_inputs()[0].name
    all_outputs = sess.run(None, {input_name: inp})
    all_out = []
    for batch_idx in range(len(all_outputs[0])):
        outputs = [all_outputs[i][batch_idx:batch_idx + 1, ...]
                   for i in range(len(all_outputs))]
        all_out.append(outputs)
    return all_out


def _dw_get_simcc_maximum(simcc_x, simcc_y):
    _, np = _dw_deps()
    N, K, Wx = simcc_x.shape
    simcc_x = simcc_x.reshape(N * K, -1)
    simcc_y = simcc_y.reshape(N * K, -1)
    x_locs = np.argmax(simcc_x, axis=1)
    y_locs = np.argmax(simcc_y, axis=1)
    locs = np.stack((x_locs, y_locs), axis=-1).astype(np.float32)
    max_val_x = np.amax(simcc_x, axis=1)
    max_val_y = np.amax(simcc_y, axis=1)
    mask = max_val_x > max_val_y
    max_val_x[mask] = max_val_y[mask]
    vals = max_val_x
    locs[vals <= 0.] = -1
    locs = locs.reshape(N, K, 2)
    vals = vals.reshape(N, K)
    return locs, vals


def _dw_decode(simcc_x, simcc_y, simcc_split_ratio):
    keypoints, scores = _dw_get_simcc_maximum(simcc_x, simcc_y)
    keypoints /= simcc_split_ratio
    return keypoints, scores


def _dw_pose_postprocess(outputs, model_input_size, center, scale,
                         simcc_split_ratio=2.0):
    _, np = _dw_deps()
    all_key, all_score = [], []
    for i in range(len(outputs)):
        simcc_x, simcc_y = outputs[i]
        keypoints, scores = _dw_decode(simcc_x, simcc_y, simcc_split_ratio)
        keypoints = keypoints / model_input_size * scale[i] + \
            center[i] - scale[i] / 2
        all_key.append(keypoints[0])
        all_score.append(scores[0])
    return np.array(all_key), np.array(all_score)


def _dw_inference_pose(session, out_bbox, oriImg, model_input_size=(288, 384),
                       dtype=None):
    if dtype is None:
        _, np = _dw_deps()
        dtype = np.float32
    resized_img, center, scale = _dw_pose_preprocess(
        oriImg, out_bbox, model_input_size)
    outputs = _dw_pose_inference(session, resized_img, dtype)
    keypoints, scores = _dw_pose_postprocess(
        outputs, model_input_size, center, scale)
    return keypoints, scores


# --- vendored: wholebody.py（裁剪为 onnx-only；返回 mmpose 同构的
#     (candidate, subset) 供 controlnet_aux 0.0.10 的 DWposeDetector
#     __call__ 绘图复用） ---

class _OnnxWholebody:
    """yolox 人检 + dwpose 关键点（全 CPU onnxruntime）。__call__ 输入 BGR
    图（与 controlnet_aux __call__ 的预处理衔接），输出：
    candidate (N,134,3)=x,y,score（openpose 序，含 neck 插入）；
    subset (N,134)=score。检测不到人时返回全空数组（shape 合法，
    上游解包不炸）。"""

    def __init__(self, det_model_path, pose_model_path):
        import onnxruntime as ort  # 懒加载
        self.det = ort.InferenceSession(
            det_model_path, providers=["CPUExecutionProvider"])
        self.pose = ort.InferenceSession(
            pose_model_path, providers=["CPUExecutionProvider"])
        ishape = self.pose.get_inputs()[0].shape  # (1,3,H,W)
        self.pose_input_size = (int(ishape[-1]), int(ishape[-2]))  # (W,H)

    def __call__(self, oriImg):
        import numpy as np
        empty = (np.zeros((0, 134, 2), dtype=np.float32),
                 np.zeros((0, 134), dtype=np.float32))
        det_result = _dw_inference_detector(self.det, oriImg,
                                            detect_classes=(0,))
        if det_result is None or det_result.shape[0] == 0:
            print("[preprocess.openpose] dwpose: 未检到人", flush=True)
            return empty
        keypoints, scores = _dw_inference_pose(
            self.pose, det_result, oriImg, self.pose_input_size)
        keypoints_info = np.concatenate(
            (keypoints, scores[..., None]), axis=-1)
        # neck = 双肩均值（openpose 第 18 点，插入索引 17）
        neck = np.mean(keypoints_info[:, [5, 6]], axis=1)
        neck[:, 2:4] = np.logical_and(
            keypoints_info[:, 5, 2:4] > 0.3,
            keypoints_info[:, 6, 2:4] > 0.3).astype(int)
        new_keypoints_info = np.insert(keypoints_info, 17, neck, axis=1)
        mmpose_idx = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
        openpose_idx = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
        new_keypoints_info[:, openpose_idx] = \
            new_keypoints_info[:, mmpose_idx]
        keypoints_info = new_keypoints_info
        # 注意：mmpose 版 Wholebody 返回的 candidate 只有 (x,y) 两列
        # （score 单列走 subset）——controlnet_aux 的 draw_handpose 按
        # (x,y) 二元组解包，给 3 列会炸 "too many values to unpack"
        candidate = keypoints_info[..., :2].astype(np.float32)
        subset = keypoints_info[..., 2].astype(np.float32)
        return candidate, subset


_dwpose_detector = None


def _get_dwpose():
    """DWPose 检测器进程级单例：controlnet_aux 空壳 + ONNX Wholebody
    （yolox_l.onnx ~207MB + dw-ll_ucoco_384.onnx ~134MB，CPU 推理 ~10-30s/
    张，不值得为它动显存）。"""
    global _dwpose_detector
    if _dwpose_detector is None:
        from controlnet_aux import DWposeDetector  # 懒加载
        d = os.path.join(PREPROC_DIR, "dwpose")
        det_p = os.path.join(d, "yolox_l.onnx")
        pose_p = os.path.join(d, "dw-ll_ucoco_384.onnx")
        for p in (det_p, pose_p):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"preprocess.openpose(detector=dwpose): 权重不存在：{p}"
                    f"（需 yolox_l.onnx + dw-ll_ucoco_384.onnx，"
                    f"来自 hf yzd-v/DWPose）")
        det = DWposeDetector.__new__(DWposeDetector)  # 绕过 mmpose 构造
        det.pose_estimation = _OnnxWholebody(det_p, pose_p)
        _dwpose_detector = det
        print(f"[preprocess.openpose] dwpose 检测器已加载：{d}", flush=True)
    return _dwpose_detector


def preprocess_openpose(image, include_hand=False, include_face=False,
                        detector="openpose"):
    """姿态骨架检测：IMAGE(真人照片) → 黑底彩色骨架图 IMAGE。
    detector='openpose'（默认，标准 OpenPose，正/背面可靠）或 'dwpose'
    （DWPose ONNX，侧身回头/遮挡/交叉肢体显著更准，自带 face+hand 关键点，
    include_hand/include_face 参数仅对 openpose 后端生效）。"""
    img = _as_image(image, "preprocess.openpose")
    detector = str(detector or "openpose").strip().lower()
    if detector == "dwpose":
        det = _get_dwpose()
        out = det(img.convert("RGB"))
    else:
        det = _get_openpose()
        out = det(img.convert("RGB"),
                  include_hand=bool(include_hand),
                  include_face=bool(include_face))
    if not isinstance(out, Image.Image):
        import numpy as np
        out = Image.fromarray(np.asarray(out))
    print(f"[preprocess.openpose] {img.size} -> {out.size} "
          f"detector={detector} hand={bool(include_hand)} "
          f"face={bool(include_face)}", flush=True)
    return {"image": out.convert("RGB")}


_midas_detector = None


def _get_midas():
    """MiDaS 检测器进程级单例（dpt_hybrid 权重 ~470MB，CPU 推理一张图几秒，
    不值得为它占显存——8GB 卡上推理期显存要留给 UNet/VAE）。"""
    global _midas_detector
    if _midas_detector is None:
        from controlnet_aux import MidasDetector  # 懒加载
        path = os.path.join(PREPROC_DIR, "midas")
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"preprocess.depth: 预处理器权重目录不存在：{path}"
                f"（需 dpt_hybrid-midas-501f0c75.pt）")
        # 目录路径模式：from_pretrained 直接 join(dir, 默认文件名)，零网络
        _midas_detector = MidasDetector.from_pretrained(
            path, local_files_only=True)
        print(f"[preprocess.depth] 检测器已加载：{path}", flush=True)
    return _midas_detector


def _release_midas():
    """释放 MiDaS 单例（~470MB RAM）。16GB 内存盒上 Flux 链路 RAM 峰值贴近
    上限，深度图产出后检测器即完成使命，常驻只会顶 OOM（exec 1049/1052/1054
    连环 exit 137 教训）；重载仅需从本地 .pt 读几秒。"""
    global _midas_detector
    if _midas_detector is not None:
        _midas_detector = None
        import gc
        gc.collect()
        print("[preprocess.depth] 检测器已释放（~470MB）", flush=True)


def preprocess_depth(image, detect_resolution=512, keep_loaded=False):
    """MiDaS 深度估计：IMAGE → 灰度深度图 IMAGE（近亮远暗，ControlNet depth
    提示图惯例），输出尺寸恒等于输入尺寸。detect_resolution 为检测内部短边
    分辨率（默认 512，细节不足可调高）——MiDaS 模型自身输入固定 384，
    该参数只影响前端预缩放。keep_loaded=True 时检测器保留常驻（批量连续
    调用省时），默认 False 用完即释放以省内存。"""
    img = _as_image(image, "preprocess.depth")
    det = _get_midas()
    det_res = max(64, int(detect_resolution))
    try:
        out = det(img.convert("RGB"), detect_resolution=det_res,
                  image_resolution=det_res)
    finally:
        if not keep_loaded:
            _release_midas()
    if not isinstance(out, Image.Image):
        import numpy as np
        out = Image.fromarray(np.asarray(out))
    # MidasDetector 输出尺寸由 image_resolution 决定（resize_image 语义），
    # 统一回缩到输入尺寸，保证下游 sd.controlnet.apply 的 hint 与出图尺寸对齐
    if out.size != img.size:
        out = out.resize(img.size, Image.BILINEAR)
    print(f"[preprocess.depth] {img.size} detect_res={det_res}", flush=True)
    return {"image": out.convert("RGB")}


def register(registry):
    registry = _VersionedRegistry(registry, OP_VERSION)
    registry.register(
        "preprocess.canny",
        inputs={"image": "IMAGE", "low_threshold": "INT",
                "high_threshold": "INT"},
        outputs={"image": "IMAGE"},
        description="Canny 边缘检测：IMAGE → 黑底白线线稿 IMAGE（ControlNet "
                    "canny 提示图）；low/high_threshold 为 cv2 双阈值 "
                    "（默认 100/200）")(preprocess_canny)
    registry.register(
        "preprocess.openpose",
        inputs={"image": "IMAGE", "include_hand": "BOOL",
                "include_face": "BOOL", "detector": "STRING"},
        outputs={"image": "IMAGE"},
        description="姿态骨架检测：真人照片 → 黑底彩色骨架图（ControlNet "
                    "openpose 提示图）；detector=openpose（默认，正/背面）/"
                    "dwpose（侧身回头/遮挡更准，ONNX 后端，权重 "
                    "/models/preprocessors/dwpose）")(preprocess_openpose)
    registry.register(
        "preprocess.depth",
        inputs={"image": "IMAGE", "detect_resolution": "INT",
                "keep_loaded": "BOOL"},
        outputs={"image": "IMAGE"},
        description="MiDaS dpt_hybrid 深度估计：IMAGE → 灰度深度图（近亮远暗，"
                    "ControlNet depth 提示图），输出尺寸=输入尺寸；"
                    "keep_loaded=True 保留检测器常驻（默认 False 用完即放）；"
                    "detect_resolution 默认 512，权重从 "
                    "/models/preprocessors/midas 本地加载")(preprocess_depth)
