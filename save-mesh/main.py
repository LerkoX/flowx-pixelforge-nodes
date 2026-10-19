"""save-mesh：从推理服务下载 3D 网格（GLB/OBJ）保存到本地目录（对标 save-video/save-image）。

下载端点 /ext/meshes/{oid} 由 tripo-sr 节点包经插件契约 v2 自注册（服务端核心零改动）。
文件须落在 server 媒体白名单目录（默认 ~/flowx-output）内画布才可见。仅本地执行器运行。
"""
import os
import re
import time

from flowx_client import emit, get_bytes, param, token


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def main():
    url = param("service_url").rstrip("/")
    mesh_id = param("mesh")
    fmt = sanitize(param("format", "glb")).lstrip(".")
    prefix = sanitize(param("filename_prefix", "flowx-mesh"))
    out_dir = os.path.expanduser(param("output_dir", "~/flowx-output"))
    tok = token()

    data = get_bytes(url, f"/ext/meshes/{mesh_id}", tok, timeout=600)
    # 以服务端实际字节为准定后缀（GLB 魔数 glTF），防参数与对象不一致
    real_ext = ".glb" if data[:4] == b"glTF" else f".{fmt or 'obj'}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir,
                        f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}{real_ext}")
    with open(path, "wb") as f:
        f.write(data)
    print(f"[save] mesh={mesh_id} -> {path} ({len(data)} bytes)", flush=True)

    emit(file_path=path, size_bytes=len(data))


if __name__ == "__main__":
    main()
