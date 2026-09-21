"""preprocess-canny：canny 线稿提取（真 cv2.Canny，服务端实现）。

约定（第三档批次 2 起）：preprocess.canny 的服务端实现以推理服务
plugins/preprocess_ops.py 为唯一来源，本节点包不再自注册该算子的实现，
仅做存在性校验 + 调用。算子缺失时给出明确报错（联系服务端部署插件）。
"""
from flowx_client import call_op, emit, get_json, param, ref, token


def _check_op(url, tok):
    ops = get_json(url, "/ops", tok, timeout=30).get("ops", [])
    if not any(o.get("name") == "preprocess.canny" for o in ops):
        raise RuntimeError(
            "算子 'preprocess.canny' 未注册：服务端插件 preprocess_ops.py "
            "未部署（该算子实现以服务端为唯一来源，节点不再自注册）")


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    _check_op(url, tok)

    out = call_op(url, "preprocess.canny",
                  {"image": ref(param("image")),
                   "low_threshold": param("low_threshold", 100, cast=int),
                   "high_threshold": param("high_threshold", 200, cast=int)},
                  tok, timeout=600)
    print(f"[canny] image={param('image')} -> edges={out['image']}", flush=True)
    emit(image=out["image"])


if __name__ == "__main__":
    main()
