"""tripo-sr：TripoSR 单图 3D 重建节点（IMAGE -> MESH）。

走任务式（/jobs + 轮询，理由同 detail-refine：GPU 串行队列可能排长任务之后）。
算子与下载端点（/ext/meshes/{oid}）由 server_op.py 经插件契约 v2 自注册。
输入图建议先去背（birefnet 抠图节点）——TripoSR 对带背景图会去背失败报错。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "mesh.triposr", tok=tok)

    t0 = time.time()
    jid = submit_job(url, {
        "name": "mesh.triposr",
        "inputs": {"image": ref(param("image")),
                   "foreground_ratio": param("foreground_ratio", 0.85, cast=float),
                   "mc_resolution": param("mc_resolution", 256, cast=int),
                   "format": param("format", "glb"),
                   "vram": param("vram", "keep")},
    }, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    took = time.time() - t0
    print(f"[tripo-sr] image={param('image')} -> mesh={out['mesh']} "
          f"verts={out['num_vertices']} faces={out['num_faces']} "
          f"({took:.1f}s)", flush=True)
    # download_url 供下游（save-mesh）与 widget 直接用；oid 本体也输出
    emit(mesh=out["mesh"],
         num_vertices=str(out["num_vertices"]),
         num_faces=str(out["num_faces"]),
         format=out["format"],
         download_url=f"{pbase}/ext/meshes/{out['mesh']}")


if __name__ == "__main__":
    main()
