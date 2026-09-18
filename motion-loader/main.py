"""motion-loader：把 MotionAdapter 组合进 SD1.x 底模 → AnimateDiffPipeline（文生视频），
输出 model/clip/vae 对象引用，下游接线与 checkpoint-loader 完全一致。

分钟级任务（首次组合要读盘+复制 UNet，数分钟）：走异步任务通道
POST /jobs + 轮询，避免同步 /op 被隧道/代理的空闲超时掐断。"""
import time

from flowx_client import emit, param, submit_job, token, wait_job


def main():
    url = param("service_url").rstrip("/")
    ckpt = param("ckpt_name")
    motion = param("motion_name")
    tok = token()

    t0 = time.time()
    jid = submit_job(url, {"name": "motion.load",
                           "inputs": {"ckpt": ckpt, "motion": motion}}, tok)
    print(f"[motion] job={jid} submitted ({ckpt} + {motion})", flush=True)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int),
                      poll=param("poll_interval", 5, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[motion] {ckpt} + {motion} composed ({time.time()-t0:.1f}s) "
          f"-> model={out['model']}", flush=True)

    emit(model_ref=out["model"], clip_ref=out["clip"], vae_ref=out["vae"])


if __name__ == "__main__":
    main()
