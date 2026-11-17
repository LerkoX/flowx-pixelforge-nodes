"""wan-vace-standalone：Wan VACE（独立装配，不经 wan-pipe）。

为什么要有这个节点（真机教训 2026-11-16）：
  同一份 VACE 权重，diffusers 官方最简脚本
  （`WanVACEPipeline(...) + pipe.to("cuda")`）在 3090 上出片完全正常；
  而经 wan-pipe 组件缓存 + offload hook 装配的 pipe，无论怎么调参
  （1.3B/14B、euler/unipc、steps 6~20、cfg 1~5）都输出彩色噪声。
  Animate 当初也是同一现象，靠「独立装配」解决 —— 这里照抄同一思路。

连法：
  wan-transformer-load ─┐
  wan-te-load ──────────┼→ wan-vace-standalone → save-video
  wan-vae-load ─────────┘     ↑
  load-video / video-person-mask / load-image
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "wan.vace_standalone", tok=tok)

    tr_id = param("transformer", "")
    te_id = param("text_encoder", "")
    vae_id = param("vae", "")
    video_id = param("video", "")
    if not tr_id:
        raise RuntimeError("wan-vace-standalone: transformer 必填"
                           "（接 wan-transformer-load）")
    if not te_id:
        raise RuntimeError("wan-vace-standalone: text_encoder 必填"
                           "（接 wan-te-load）")
    if not vae_id:
        raise RuntimeError("wan-vace-standalone: vae 必填"
                           "（接 wan-vae-load）")
    if not video_id:
        raise RuntimeError("wan-vace-standalone: video（原视频）必填")

    inputs = {
        "transformer": ref(tr_id),
        "text_encoder": ref(te_id),
        "vae": ref(vae_id),
        "video": ref(video_id),
        "prompt": param("prompt", ""),
        "negative_prompt": param("negative_prompt", ""),
        "conditioning_scale": param("conditioning_scale", 1.0, float),
        "flow_shift": param("flow_shift", 5.0, float),
        "scheduler": param("scheduler", "euler"),
        "num_frames": param("num_frames", 0, int),
        "height": param("height", 0, int),
        "width": param("width", 0, int),
        "steps": param("steps", 20, int),
        "guidance_scale": param("guidance_scale", 5.0, float),
        "seed": param("seed", -1, int),
    }
    mask_id = param("mask", "")
    if mask_id:
        inputs["mask"] = ref(mask_id)
    ref_id = param("reference_image", "")
    if ref_id:
        inputs["reference_image"] = ref(ref_id)

    t0 = time.time()
    jid = submit_job(url, {"name": "wan.vace_standalone", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[vace-standalone] job={jid} steps={inputs['steps']} "
          f"cfg={inputs['guidance_scale']} shift={inputs['flow_shift']} "
          f"sched={inputs['scheduler']} mask={'yes' if mask_id else 'no'} "
          f"ref={'yes' if ref_id else 'no'}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    fps = out.get("fps") or 16
    print(f"[vace-standalone] -> video={out['video']} seed={out['seed']} "
          f"fps={fps} ({time.time()-t0:.1f}s)", flush=True)
    emit(video=out["video"], seed=out["seed"], fps=fps)


if __name__ == "__main__":
    main()
