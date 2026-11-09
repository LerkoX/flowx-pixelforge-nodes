"""minimax-h3-gen：MiniMax-H3 联合视频+音频生成（fl2va 首末帧 / t2va 文生）。

自包含节点：服务端算子 minimax.h3.gen 由 server_op.py 自注册（ensure_plugin）。
guidance-distilled：无负提示、无 cfg；24fps 固定，5~15 秒（num_frames 自动对齐
17n+5）；挂 turbo LoRA 时 4 步出片。输出 VIDEO 对象（含 32kHz 立体声音轨，
服务端落盘时自动 mux 进 mp4）。

分钟级任务：走异步任务通道 POST /jobs + 轮询（同 video-gen 模式）。
"""
import json
import time

from flowx_client import (emit, emit_preview, emit_result_preview,
                          ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()

    ensure_plugin(url, "minimax.h3.gen", tok=tok)

    inputs = {
        "model": ref(param("model_ref")),
        "prompt": param("prompt"),
        "num_frames": param("num_frames", 125, int),
        "seed": param("seed", -1, int),
        "steps": param("steps", 5, int),
        "preview_every": 0,
    }
    # 画布自定义（0=官方默认 16:9 768p）
    h = param("height", 0, int)
    w = param("width", 0, int)
    if h > 0:
        inputs["height"] = h
    if w > 0:
        inputs["width"] = w
    image_id = param("image", "")
    if image_id:
        inputs["image"] = ref(image_id)
    last_id = param("last_image", "")
    if last_id:
        inputs["last_image"] = ref(last_id)
    # ref2va：references JSON 原样透传（服务端校验格式与对象类型）
    refs = param("references", "")
    # reference_image：单张图对象 ID（流水线友好）——自动包成 references JSON。
    # 为什么需要它：references 是字符串 JSON，画布上无法把 LoadImage 的输出接进
    # JSON 字符串里（模板只做整值替换，不做字符串内嵌），故给单图场景一条直路。
    ref_img = param("reference_image", "")
    if not refs.strip() and ref_img.strip():
        refs = json.dumps([{"type": "image", "id": ref_img.strip()}])
        print(f"[minimax-h3-gen] reference_image -> references: {refs}", flush=True)
    if refs.strip():
        inputs["references"] = refs
    # 局部编辑（人物替换）：源视频对象 + SAM 遮罩对象（video-person-mask）
    ipv = param("inpaint_video", "")
    ipm = param("inpaint_mask", "")
    if ipv or ipm:
        if not (ipv and ipm):
            raise RuntimeError("inpaint_video 与 inpaint_mask 必须同时接线")
        inputs["inpaint_video"] = ref(ipv)
        inputs["inpaint_mask"] = ref(ipm)
        # 源约束强度旋钮：soft（遮罩内生成权重）/ lock（前期全图锁源占比）
        inputs["inpaint_soft"] = float(param("inpaint_soft", "1") or "1")
        inputs["inpaint_lock"] = float(param("inpaint_lock", "0") or "0")
        inputs["inpaint_aug"] = float(param("inpaint_aug", "0.999") or "0.999")
        inputs["mask_schedule"] = param("mask_schedule", "linear")
        inputs["window_start"] = float(param("window_start", "0") or "0")
        inputs["window_end"] = float(param("window_end", "1") or "1")
    mode = ("inpaint" if (ipv and ipm)
            else "ref2va" if refs.strip()
            else ("fl2va" if (image_id or last_id) else "t2va"))
    print(f"[minimax-h3-gen] mode={mode} frames={inputs["num_frames"]} "
          f"steps={inputs['steps']} seed={inputs['seed']}", flush=True)

    t0 = time.time()
    jid = submit_job(url, {"name": "minimax.h3.gen", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[minimax-h3-gen] job={jid} submitted", flush=True)

    result = wait_job(url, jid, tok,
                      timeout=param("job_timeout", 7200, int),
                      poll=param("poll_interval", 5, int))
    outs = result["outputs"]
    seed = outs["seed"].get("value")
    video_id = outs["video"].get("id")
    print(f"[minimax-h3-gen] done in {time.time()-t0:.1f}s "
          f"-> video={video_id} seed={seed}", flush=True)
    # 完成后上报结果预览（widget 识别 /videos/ 路径用 <video> 内嵌播放）
    if video_id:
        emit_result_preview(pbase, video_id, tok, kind="videos")
    emit(video=video_id, seed=seed)


if __name__ == "__main__":
    main()
