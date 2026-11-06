"""beautify-batch：目录 → 逐张「智能裁方 + 美白磨皮 + 大眼瘦脸」→ 对象 ID 列表。

美化在服务端算子 beautify.auto 完成（cv2/numpy/insightface，本机零重依赖）；
节点只做目录扫描、上传、逐张调算子、收集结果（Studio 无循环，批量收进单节点）。
输出 images 与 grid-compose 直接对接。
"""
import os

from flowx_client import (call_op, emit, emit_result_preview, ensure_plugin,
                          param, ref, token)
from image_batch import collect, dumps, load_batch, loads

OP = "beautify.auto"


def op_kwargs():
    """从节点参数组装算子入参（抽出来便于单测）。"""
    return dict(crop=param("crop_size", 800, cast=int),
                whiten=param("whiten", 0.5, cast=float),
                smooth=param("smooth", 0.4, cast=float),
                eye=param("eye", 0.12, cast=float),
                slim=param("slim", 0.10, cast=float),
                det_thresh=param("det_thresh", 0.3, cast=float))


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    plugin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_op.py")
    actual = ensure_plugin(url, OP, plugin_path=plugin, tok=tok)
    files, path = collect(param("dir"), pattern=param("pattern", "*.png;*.jpg;*.jpeg"),
                          sort=param("sort", "name"), reverse=param("reverse", "false"),
                          limit=param("limit", 0, cast=int),
                          offset=param("offset", 0, cast=int),
                          recursive=param("recursive", "false"))
    if not files:
        raise RuntimeError(f"beautify-batch: 目录无匹配文件 {path}")
    ids_json, names_json = load_batch(url, files, tok)
    ids, names = loads(ids_json), loads(names_json)
    kw = op_kwargs()
    out_ids = []
    for i, (oid, name) in enumerate(zip(ids, names), 1):
        r = call_op(url, actual, {"image": ref(oid), **kw}, tok)
        out_ids.append(r["image"])
        print(f"[beautify-batch] {i}/{len(ids)} {name} -> {r['image']}", flush=True)
    print(f"[beautify-batch] {len(out_ids)} 张完成", flush=True)
    emit(images=dumps(out_ids), count=len(out_ids), names=names_json, dir=path)
    emit_result_preview(url, out_ids[-1], tok)


if __name__ == "__main__":
    main()
