"""h3-text-encode：MiniMax-H3 文本编码（prompt → COND）。

对齐 SD 的 clip-text-encode：把编码从 gen 里拆出来，好处
①同一 prompt 多次生成复用编码 ②可接 cond 类算子（combine/set-area 等）
③开投影 TE 时编码只发生一次。

按 image/last_image/references 自动分派（服务端决定 mode）：
  t2va  （纯文本）/ fl2va（首末帧）/ ref2va（图/视频/音频参考）
image/last_image 原样透传 —— gen 还要用它们做 keyframe latent 编码。

分钟级任务走 /jobs 异步（编码同步在 GPU 串行队列里可能排队很久）。
"""
from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "minimax.h3.encode", tok=tok)

    model_ref = param("model_ref")
    prompt = param("prompt")
    inputs = {
        "model": ref(model_ref),
        "prompt": prompt,
        "height": param("height", 0, int),
        "width": param("width", 0, int),
    }
    img = param("image", "")
    if img:
        inputs["image"] = ref(img)
    last = param("last_image", "")
    if last:
        inputs["last_image"] = ref(last)
    refs = param("references", "")
    if refs.strip():
        inputs["references"] = refs.strip()

    mode = ("ref2va" if inputs.get("references")
            else "fl2va" if ("image" in inputs or "last_image" in inputs)
            else "t2va")
    prev = prompt if len(prompt) <= 60 else prompt[:60] + "..."
    print(f"[h3-text-encode] mode={mode} prompt=\"{prev}\"", flush=True)

    jid = submit_job(url, {"name": "minimax.h3.encode", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 1800, int), poll=5)
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[h3-text-encode] done -> cond={out['cond']}", flush=True)
    # 三个输出：cond 给 gen，image/last_image 透传给 gen（画布上不用重复连线）
    emit(cond=out["cond"], image=out.get("image", ""),
         last_image=out.get("last_image", ""))


if __name__ == "__main__":
    main()
