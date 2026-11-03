"""zero123：Stable Zero123 单图新视角合成节点（IMAGE -> IMAGE）。

走任务式（/jobs + 轮询，理由同 tripo-sr：GPU 串行队列可能排长任务之后）。
算子 novelview.zero123 由 server_op.py 经插件契约自注册。
输入图建议先去背（matting-birefnet 节点输出 RGBA 最佳——算子会合成白底）。
"""
import time

from flowx_client import (emit, emit_input_media, emit_result_preview,
                          ensure_plugin, host_base, param, ref, submit_job,
                          token, wait_job)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "novelview.zero123", tok=tok)

    # 输入原图物化：画布对比控件的「原图」侧
    emit_input_media(url, param("image"), tok)

    t0 = time.time()
    jid = submit_job(url, {
        "name": "novelview.zero123",
        "inputs": {"image": ref(param("image")),
                   "elevation": param("elevation", 0.0, cast=float),
                   "azimuth": param("azimuth", 0.0, cast=float),
                   "distance": param("distance", 0.0, cast=float),
                   "steps": param("steps", 50, cast=int),
                   "guidance_scale": param("guidance_scale", 3.0, cast=float),
                   "vram": param("vram", "after_op")},
    }, tok)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    took = time.time() - t0
    print(f"[zero123] image={param('image')} elevation={param('elevation')} "
          f"azimuth={param('azimuth')} -> image={out['image']} ({took:.1f}s)",
          flush=True)
    emit(image=out["image"],
         width=str(out.get("width", 256)),
         height=str(out.get("height", 256)))
    emit_result_preview(url, out["image"], tok)


if __name__ == "__main__":
    main()
