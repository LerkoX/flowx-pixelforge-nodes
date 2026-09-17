"""motion-loader：把 MotionAdapter 组合进 SD1.x 底模 → AnimateDiffPipeline（文生视频），
输出 model/clip/vae 对象引用，下游接线与 checkpoint-loader 完全一致。"""
import time

from flowx_client import call_op, emit, param, token


def main():
    url = param("service_url").rstrip("/")
    ckpt = param("ckpt_name")
    motion = param("motion_name")
    tok = token()

    t0 = time.time()
    out = call_op(url, "motion.load", {"ckpt": ckpt, "motion": motion},
                  tok, timeout=1800)
    print(f"[motion] {ckpt} + {motion} composed ({time.time()-t0:.1f}s) "
          f"-> model={out['model']}", flush=True)

    emit(model_ref=out["model"], clip_ref=out["clip"], vae_ref=out["vae"])


if __name__ == "__main__":
    main()
