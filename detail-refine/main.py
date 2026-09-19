"""detail-refine：ADetailer 式局部重绘（脸/手检测 → 裁剪放大 → img2img → 羽化贴回）。
首个自包含节点：server_op.py 随节点包自注册到推理服务（ensure_plugin）。"""
from flowx_client import call_op, emit, ensure_plugin, param, ref, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    detector = param("detector", "face")

    ensure_plugin(url, "detail.refine", tok=tok)

    out = call_op(url, "detail.refine",
                  {"model": ref(param("model_ref")),
                   "pos": ref(param("positive")),
                   "neg": ref(param("negative")),
                   "image": ref(param("image")),
                   "detector": detector,
                   "conf": param("conf", 0.3, cast=float),
                   "padding": param("padding", 0.4, cast=float),
                   "denoise": param("denoise", 0.4, cast=float),
                   "steps": param("steps", 20, cast=int),
                   "cfg": param("cfg", 7.0, cast=float),
                   "sampler_name": param("sampler_name", "euler"),
                   "seed": param("seed", -1, cast=int),
                   "guide_size": param("guide_size", 512, cast=int),
                   "max_targets": param("max_targets", 4, cast=int),
                   "feather": param("feather", 16, cast=int)},
                  tok, timeout=1800)
    print(f"[detail-refine] detector={detector} refined={out.get('count', 0)} "
          f"image={param('image')} -> {out['image']}", flush=True)
    emit(image=out["image"], count=str(out.get("count", 0)))


if __name__ == "__main__":
    main()
