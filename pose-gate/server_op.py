"""pose-gate 节点的服务端算子插件（随节点包自注册，经 POST /admin/plugins 上传）。

姿态检测 + 人工确认闸门：
- pose.detect：DWPose（ONNX 后端）全身检测 → 骨架图 IMAGE + 关键点 JSON
  （STRING，归一化坐标，供节点 widget 画布内拖拽修正）。
- pose.render：关键点 JSON → 重渲染骨架图（widget 改点后生成新 hint）。
- 契约 v2 路由 /ext/gate/{key}：闸门会话（节点创建 → widget 查询/提交决策
  → 节点轮询取决策），内存态 + TTL，服务重启即失效（节点侧超时兜底）。

DWPose 检测代码移植自 preprocess 算子族（同一份 vendored ONNX Wholebody：
yolox_l 人检 + dw-ll_ucoco_384 关键点，CPU 推理），绘图复用 controlnet_aux
的 dwpose util（draw_bodypose/draw_handpose/draw_facepose，均吃归一化坐标）。
与 preprocess.openpose(detector=dwpose) 的差异：本算子返回关键点且
include_hand/include_face 对 dwpose 真正生效（旧算子对 dwpose 忽略开关，
face/hand 恒画）。
"""
import json
import os
import time

from PIL import Image

PREPROC_DIR = os.environ.get("PREPROCESSOR_MODELS_DIR", "/models/preprocessors")

# ===================== DWPose（ONNX 后端，vendored） =====================
# 移植自 nodes/_common/server_ops/preprocess.py（同源：MIT
# Fannovel16/comfyui_controlnet_aux 的 dw_onnx/cv_ox_det.py、cv_ox_pose.py、
# wholebody.py，裁剪为 onnxruntime-only）。权重：
# /models/preprocessors/dwpose/{yolox_l.onnx, dw-ll_ucoco_384.onnx}


def _dw_deps():
    import cv2  # 懒加载（本地无 cv2 环境需可导入本文件）
    import numpy as np
    return cv2, np


def _dw_nms(boxes, scores, nms_thr):
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


def _dw_inference_detector(session, oriImg, detect_classes=(0,), dtype=None):
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
    dst_dir = np.array([0., dst_h * -0.5])
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


class _OnnxWholebody:
    """yolox 人检 + dwpose 关键点（全 CPU onnxruntime）。__call__ 输入 BGR
    图，输出 candidate (N,134,3)=x,y,score（openpose 序，含 neck 插入索引 17）
    与 subset (N,134)=score。检测不到人时返回全空数组。"""

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
            print("[pose.detect] dwpose: 未检到人", flush=True)
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
        candidate = keypoints_info[..., :2].astype(np.float32)
        subset = keypoints_info[..., 2].astype(np.float32)
        return candidate, subset


_wholebody = None


def _get_wholebody():
    """ONNX Wholebody 进程级单例（yolox_l.onnx ~207MB +
    dw-ll_ucoco_384.onnx ~134MB，CPU 推理 ~10-30s/张）。"""
    global _wholebody
    if _wholebody is None:
        d = os.path.join(PREPROC_DIR, "dwpose")
        det_p = os.path.join(d, "yolox_l.onnx")
        pose_p = os.path.join(d, "dw-ll_ucoco_384.onnx")
        for p in (det_p, pose_p):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"pose.detect: 权重不存在：{p}"
                    f"（需 yolox_l.onnx + dw-ll_ucoco_384.onnx）")
        _wholebody = _OnnxWholebody(det_p, pose_p)
        print(f"[pose.detect] dwpose 检测器已加载：{d}", flush=True)
    return _wholebody


# ===================== 骨架渲染（detect 与 render 共用同一路径） =====================

def _render_kp(candidate, subset, width, height, include_hand, include_face):
    """归一化关键点（单人 (134,2)/(134,)）→ 黑底彩色骨架 PIL 图。

    可见性规则与 controlnet_aux DWposeDetector.__call__ 一致
    （score<=0.3 的点不画）；include_hand/include_face 控制手/脸是否绘制。
    """
    import numpy as np
    from controlnet_aux.dwpose.util import (draw_bodypose, draw_facepose,
                                            draw_handpose)
    W, H = int(width), int(height)
    cand = np.asarray(candidate, dtype=np.float32).reshape(134, 2).copy()
    sub = np.asarray(subset, dtype=np.float32).reshape(134).copy()
    cand[sub < 0.3] = -1
    body = cand[:18]
    score = sub[:18].copy()
    for j in range(18):
        score[j] = float(j) if score[j] > 0.3 else -1.0
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    canvas = draw_bodypose(canvas, body, score[None])
    if include_hand:
        canvas = draw_handpose(canvas, [cand[92:113], cand[113:134]])
    if include_face:
        canvas = draw_facepose(canvas, cand[24:92][None])  # 需带 person 轴
    return Image.fromarray(canvas)


def pose_detect(image, include_hand=True, include_face=False):
    """DWPose 姿态检测：IMAGE → 骨架图 IMAGE + 关键点 JSON STRING
    （归一化坐标 134 点：0-17 body、18-23 foot、24-91 face、92-112 左手、
    113-133 右手）+ 骨架画布宽高。多人时取 body 平均分最高的一人。
    include_hand/include_face 控制骨架图是否画手/脸（widget 决策时可改）。"""
    import cv2
    import numpy as np
    from controlnet_aux.util import HWC3, resize_image
    if not isinstance(image, Image.Image):
        raise ValueError(
            f"pose.detect: image 需为 IMAGE 对象，got {type(image).__name__}")
    bgr = cv2.cvtColor(np.asarray(image.convert("RGB"), dtype=np.uint8),
                       cv2.COLOR_RGB2BGR)
    bgr = resize_image(HWC3(bgr), 512)  # 与 DWposeDetector.__call__ 同口径
    H, W = bgr.shape[:2]
    candidate, subset = _get_wholebody()(bgr)
    if candidate.shape[0] == 0:
        raise RuntimeError("pose.detect: 未检测到人物，无法生成骨架")
    # 多人取 body（前 18 点）平均分最高者
    best = int(np.argmax(subset[:, :18].mean(axis=1)))
    cand = candidate[best].copy()      # (134,2) px
    sub = subset[best].copy()          # (134,)
    cand[:, 0] /= float(W)
    cand[:, 1] /= float(H)
    img = _render_kp(cand, sub, W, H, bool(include_hand), bool(include_face))
    kp = {"width": W, "height": H,
          "candidate": [[round(float(x), 5), round(float(y), 5)]
                        for x, y in cand],
          "subset": [round(float(s), 4) for s in sub]}
    print(f"[pose.detect] {image.size} -> {img.size} "
          f"hand={bool(include_hand)} face={bool(include_face)}", flush=True)
    return {"image": img, "keypoints": json.dumps(kp), "width": W,
            "height": H}


def pose_render(keypoints, include_hand=True, include_face=False):
    """从关键点 JSON 重渲染骨架图（widget 人工修正后生成新 ControlNet hint）。
    keypoints 为 pose.detect 的 JSON（允许 candidate/subset 被改）。"""
    kp = json.loads(keypoints)
    img = _render_kp(kp["candidate"], kp["subset"], kp["width"],
                     kp["height"], bool(include_hand), bool(include_face))
    print(f"[pose.render] {img.size} hand={bool(include_hand)} "
          f"face={bool(include_face)}", flush=True)
    return {"image": img}


# ===================== 闸门会话（契约 v2 路由） =====================

_GATES = {}
GATE_TTL = 3600  # 秒；超时未取/未决策的会话惰性清除


def _purge():
    now = time.time()
    for k in [k for k, v in _GATES.items() if now - v["created"] > GATE_TTL]:
        _GATES.pop(k, None)


def register(registry):
    registry.register(
        "pose.detect",
        inputs={"image": "IMAGE", "include_hand": "BOOL",
                "include_face": "BOOL"},
        outputs={"image": "IMAGE", "keypoints": "STRING", "width": "INT",
                 "height": "INT"},
        description="DWPose 姿态检测 + 关键点导出：真人照片 → 黑底彩色骨架图 + "
                    "关键点 JSON（归一化 134 点）+ 画布宽高；供 pose-gate 节点"
                    "人工确认/修正后用 pose.render 重渲染")(pose_detect)
    registry.register(
        "pose.render",
        inputs={"keypoints": "STRING", "include_hand": "BOOL",
                "include_face": "BOOL"},
        outputs={"image": "IMAGE"},
        description="关键点 JSON → 重渲染骨架图（pose-gate widget 修正后的 "
                    "ControlNet hint 生成）")(pose_render)


def register_routes(add_route):
    """闸门会话端点：节点创建 → widget 查询/提交决策 → 节点轮询取决策。

    POST   /ext/gate/{key}          创建会话 {image_id, source_id, keypoints}
    GET    /ext/gate/{key}          查询会话（含 decision，无则 null）
    POST   /ext/gate/{key}/decision 提交决策 {action: confirm|render,
                                    keypoints?, include_hand?, include_face?}
    DELETE /ext/gate/{key}          会话清理（节点结束/被取消时）
    """
    from fastapi import Body, HTTPException

    def list_gates():
        _purge()
        return {"gates": [{"key": k, "created": v["created"]}
                          for k, v in _GATES.items()]}

    def create(key: str, payload: dict = Body(...)):
        _purge()
        _GATES[key] = {"created": time.time(),
                       "image_id": payload.get("image_id"),
                       "source_id": payload.get("source_id"),
                       "keypoints": payload.get("keypoints"),
                       "result_png": None,
                       "decision": None}
        return {"ok": True}

    def query(key: str):
        _purge()
        s = _GATES.get(key)
        if s is None:
            raise HTTPException(status_code=404, detail="gate not found")
        # 不返回 result_png（bytes 无法 JSON 序列化，会 500）
        return {"created": s["created"], "image_id": s["image_id"],
                "source_id": s["source_id"], "keypoints": s["keypoints"],
                "decision": s["decision"]}

    def decide(key: str, payload: dict = Body(...)):
        _purge()
        s = _GATES.get(key)
        if s is None:
            raise HTTPException(status_code=404,
                                detail="gate not found (expired?)")
        if s["decision"] is not None:
            return {"ok": True, "decision": s["decision"]}  # 幂等
        action = str(payload.get("action") or "confirm")
        if action == "render":
            kp = payload.get("keypoints")
            if isinstance(kp, dict):
                kp = json.dumps(kp)
            if not kp:
                raise HTTPException(status_code=400,
                                    detail="action=render 需携带 keypoints")
            img = pose_render(
                kp,
                include_hand=bool(payload.get("include_hand", True)),
                include_face=bool(payload.get("include_face", False)))["image"]
            # 结果图字节留在会话里（GET .../image 拉取），由节点侧经 POST /images
            # 入对象库——路由上下文 import 的 store 与核心端点可能不是同一实例
            # （插件命名空间隔离），直接 put 会导致 /images/{id} 取不到。
            import io
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            s["result_png"] = buf.getvalue()
            s["decision"] = {"action": "render", "image_id": None}
        else:
            s["decision"] = {"action": "confirm", "image_id": s["image_id"]}
        print(f"[pose-gate] {key} decision={s['decision']['action']}",
              flush=True)
        return {"ok": True, "decision": s["decision"]}

    def result_image(key: str):
        from fastapi.responses import Response
        s = _GATES.get(key)
        if s is None or not s.get("result_png"):
            raise HTTPException(status_code=404, detail="no result image")
        return Response(content=s["result_png"], media_type="image/png")

    def delete(key: str):
        _GATES.pop(key, None)
        return {"ok": True}

    add_route("/ext/gates", list_gates)
    add_route("/ext/gate/{key}", create, methods=("POST",))
    add_route("/ext/gate/{key}", query)
    add_route("/ext/gate/{key}/decision", decide, methods=("POST",))
    add_route("/ext/gate/{key}/image", result_image)
    add_route("/ext/gate/{key}", delete, methods=("DELETE",))
