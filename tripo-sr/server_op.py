"""tripo-sr 节点的服务端算子插件（随节点包自注册，经 POST /admin/plugins 上传）。

TripoSR 单图 3D 重建：IMAGE -> MESH（GLB/OBJ 字节，带顶点色）。
- 权重：/models/triposr（config.yaml + model.ckpt，MIT 许可，见 HF
  stabilityai/TripoSR）；容器内源码 /opt/triposr/tsr（镜像内置，打过两个补丁：
  torchmcubes→skimage fallback、rembg 懒加载）。
- 契约 v2：register_routes 自带 /ext/meshes/{oid} 下载端点（GLB 字节流），
  服务端核心零改动。
- 输入图建议先去背（BiRefNet 等抠图节点）；RGBA 带透明通道亦可（TripoSR 按
  alpha 判断是否需要去背，已透明则直接用）。
"""
import os
import sys
import threading

_TRIPOSR_DIR = os.environ.get("TRIPOSR_DIR", "/models/triposr")
_TSR_SRC = os.environ.get("TSR_SRC", "/opt/triposr")

_model = None
_lock = threading.Lock()


def _load_model():
    global _model
    with _lock:
        if _model is None:
            # DINOv2 图像编码器经 transformers 从 HF 下载（~1GB），HF_HOME 指到
            # bind-mount 持久化，防容器重建后重复下载；必须在 import 前设置。
            os.environ.setdefault("HF_HOME", "/models/hf")
            # rembg 首次用会下载 u2net 权重（~170MB），同样落 bind-mount
            os.environ.setdefault("U2NET_HOME", "/models/u2net")
            if _TSR_SRC not in sys.path:
                sys.path.insert(0, _TSR_SRC)
            import torch
            from tsr.system import TSR
            m = TSR.from_pretrained(_TRIPOSR_DIR, "config.yaml", "model.ckpt")
            _model = m.half().cuda()
            print("[mesh.triposr] TripoSR loaded (fp16/cuda)", flush=True)
    return _model


def _release():
    global _model
    with _lock:
        _model = None
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def register(registry):
    @registry.register(
        "mesh.triposr",
        inputs={"image": "IMAGE", "foreground_ratio": "FLOAT",
                "mc_resolution": "INT", "format": "STRING", "vram": "STRING"},
        outputs={"mesh": "MESH", "num_vertices": "INT", "num_faces": "INT",
                 "format": "STRING"},
        description="TripoSR 单图 3D 重建（IMAGE -> MESH）。"
                    "foreground_ratio=前景占比(0.85)；mc_resolution=marching cubes "
                    "分辨率(256，越高越细越慢)；format=glb/obj；vram=keep/after_op")
    def _op_mesh_triposr(image, foreground_ratio=0.85, mc_resolution=256,
                         format="glb", vram="keep"):
        model = _load_model()  # 内含 sys.path.insert + import tsr（必须先于下方 import）
        import numpy as np
        import torch
        from PIL import Image as _Image
        from tsr.utils import remove_background, resize_foreground
        # 对齐官方 run.py 预处理：去背（RGBA 已透明则跳过；不透明 RGB 会走 rembg，
        # 未安装时报错并指引用抠图节点）→ 前景缩放 → alpha 合成到灰底(0.5)
        img = remove_background(image)
        img = resize_foreground(img, float(foreground_ratio))
        arr = np.asarray(img).astype(np.float32) / 255.0
        if arr.shape[-1] == 4:
            arr = arr[..., :3] * arr[..., 3:4] + (1.0 - arr[..., 3:4]) * 0.5
        img = _Image.fromarray((arr * 255.0).astype(np.uint8))
        with torch.no_grad():
            codes = model([img], device="cuda")
        meshes = model.extract_mesh(codes, has_vertex_color=True,
                                    resolution=int(mc_resolution))
        mesh = meshes[0]
        fmt = str(format).lower()
        if fmt not in ("glb", "obj"):
            raise ValueError(f"mesh.triposr: format 只支持 glb/obj，收到 {format!r}")
        data = mesh.export(file_type=fmt)
        if isinstance(data, str):
            data = data.encode("utf-8")
        nv, nf = len(mesh.vertices), len(mesh.faces)
        print(f"[mesh.triposr] verts={nv} faces={nf} {fmt}={len(data)}B",
              flush=True)
        if str(vram) == "after_op":
            _release()
        return {"mesh": data, "num_vertices": nv, "num_faces": nf,
                "format": fmt}


def register_routes(add_route):
    """契约 v2：MESH 对象的下载端点（节点包自带，服务端核心不动）。

    store 在请求时才 import（插件在 main 装配期加载，顶层 import 会撞循环导入）。
    """
    def download(oid: str):
        from fastapi import HTTPException
        from fastapi.responses import Response
        from app.main import store
        try:
            entry = store.get(oid, "MESH")
        except KeyError:
            raise HTTPException(status_code=404,
                                detail=f"mesh '{oid}' not found (expired?)")
        data = entry["data"]
        if isinstance(data, str):
            data = data.encode("utf-8")
        is_glb = data[:4] == b"glTF"
        ct = "model/gltf-binary" if is_glb else "text/plain; charset=utf-8"
        ext = ".glb" if is_glb else ".obj"
        return Response(
            content=data, media_type=ct,
            headers={"Content-Disposition":
                     f'attachment; filename="mesh_{oid[:8]}{ext}"'})

    add_route("/ext/meshes/{oid}", download)
