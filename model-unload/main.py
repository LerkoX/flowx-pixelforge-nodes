"""model-unload：显式卸载常驻模型，腾显存（对标 ComfyUI 的 unload/free）。

为什么需要它（dev-plan §21.1）：淘汰原先只看"条目个数"，视频模型（SVD/AnimateDiff）
与图像 checkpoint 同池常驻时，8GB 卡会走 host memory 兜底 → 采样从 0.5s/步 退化到
17s/步、精修 127s/步。显存预算式淘汰（服务端）解决"加载时"的挤压，本节点解决
"跑完主动腾地方"这一步：视频段结束 → 卸载 → 随后图像采样回到基线。
"""
from flowx_client import emit, param, post_json, token


def main():
    url = param("service_url").rstrip("/")
    target = param("target", "")
    tok = token()
    res = post_json(url, "/model/unload", {"target": target}, tok, timeout=300)
    unloaded = res.get("unloaded", "(none)")
    resident = res.get("resident", "(empty)")
    print(f"[model-unload] target='{target or '(all)'}' "
          f"-> unloaded={unloaded} resident={resident}", flush=True)
    emit(unloaded=unloaded, resident=resident)


if __name__ == "__main__":
    main()
