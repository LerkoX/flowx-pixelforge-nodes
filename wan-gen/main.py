"""wan-gen：通义万相视频生成（T2V / I2V / 首尾帧）。

自包含节点：服务端算子 wan.gen 由 server_op.py 自注册（ensure_plugin）。

与 minimax-h3-gen 的关键差异（别照抄 H3 的参数习惯）：
- **Wan 有 CFG**：guidance_scale > 1 时每步两次前向，negative_prompt 才生效；
  H3 是 guidance-distilled，既无 CFG 也无负提示
- **帧数契约 4k+1**（81/121/161…），不是 H3 的 17n+5
- **fps**：Wan2.1 全系 16；Wan2.2-TI2V-5B 是 24（按 transformer 名自动判定）
- **无音轨**（Wan 不生成音频）

image 留空 → T2V；给 image → I2V；image + last_image → 首尾帧插值。
分钟级任务，走异步 POST /jobs + 轮询。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "wan.gen", tok=tok)

    model_id = param("model", "")
    if not model_id:
        raise RuntimeError("wan-gen: model 必填（接 wan-load 的 model 输出）")

    image_id = param("image", "")
    last_id = param("last_image", "")

    inputs = {
        "model": ref(model_id),
        "prompt": param("prompt", ""),
        "negative_prompt": param("negative_prompt", ""),
        "num_frames": param("num_frames", 81, int),
        "height": param("height", 480, int),
        "width": param("width", 832, int),
        "steps": param("steps", 30, int),
        "guidance_scale": param("guidance_scale", 5.0, float),
        "seed": param("seed", -1, int),
    }
    if image_id:
        inputs["image"] = ref(image_id)
    if last_id:
        inputs["last_image"] = ref(last_id)

    mode = ("flf2v" if (image_id and last_id)
            else "i2v" if image_id else "t2v")

    t0 = time.time()
    jid = submit_job(url, {"name": "wan.gen", "inputs": inputs}, tok)

    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[wan-gen] job={jid} mode={mode} "
          f"frames={inputs['num_frames']} {inputs['width']}x{inputs['height']} "
          f"steps={inputs['steps']} cfg={inputs['guidance_scale']} "
          f"seed={inputs['seed']}", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 3600, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    fps = out.get("fps") or 16
    print(f"[wan-gen] -> video={out['video']} seed={out['seed']} "
          f"fps={fps} ({time.time()-t0:.1f}s)", flush=True)
    emit(video=out["video"], seed=out["seed"], fps=fps)


if __name__ == "__main__":
    main()
