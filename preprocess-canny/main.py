"""preprocess-canny：canny 风格线稿提取（自包含节点，server_op.py 经 ensure_plugin 自注册）。

输入上游生成的图像（如 SD3 出图），输出黑底白线线稿，供 ControlNet canny
控制采样作 hint，实现"先生成线稿 → 再按线稿出图"的两段式流水线。
"""
from flowx_client import call_op, emit, ensure_plugin, get_json, param, ref, token


def _ensure(url, tok):
    """软注册：算子已存在（无论哪个插件文件提供）直接用，避免与服务器上
    合并部署的 preprocess_ops.py 互相覆盖；缺失时才上传自注册。"""
    ops = get_json(url, "/ops", tok, timeout=30).get("ops", [])
    if any(o.get("name") == "preprocess.canny" for o in ops):
        return
    ensure_plugin(url, "preprocess.canny", tok=tok)


def main():
    url = param("service_url").rstrip("/")
    tok = token()

    _ensure(url, tok)

    out = call_op(url, "preprocess.canny",
                  {"image": ref(param("image")),
                   "passes": param("passes", 2, cast=int),
                   "cutoff": param("cutoff", 1, cast=int)},
                  tok, timeout=600)
    print(f"[canny] image={param('image')} -> edges={out['image']}", flush=True)
    emit(image=out["image"])


if __name__ == "__main__":
    main()
