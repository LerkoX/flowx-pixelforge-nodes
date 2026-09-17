"""motion-loader：把 MotionAdapter 组合进 SD1.x 底模 → AnimateDiffPipeline（文生视频），
输出 model/clip/vae 对象引用，下游接线与 checkpoint-loader 完全一致。"""
import time

from flowx_client import call_op, emit, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    model = param("model_ref")
    motion = param("motion_name")
    tok = token()

    t0 = time.time()
    out = call_op(url, "motion.load", {"model": ref(model), "motion": motion},
                  tok, timeout=1200)
    print(f"[motion] {motion} composed ({time.time()-t0:.1f}s) -> model={out['model']}",
          flush=True)

    emit(model_ref=out["model"], clip_ref=out["clip"], vae_ref=out["vae"])


if __name__ == "__main__":
    main()
