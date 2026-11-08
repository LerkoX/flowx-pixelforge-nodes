"""video-concat：多段视频拼接为长视频（长视频工作流的合成收尾）。

逐段下载推理服务视频对象 → ffmpeg 滤镜 concat（视频重编码保证段间参数
归一）+ acrossfade 音频交叉淡化（H3 各段音频独立生成，硬接会断响）→ 落盘。
纯本地节点（ffmpeg 在 Studio 机器执行），不消耗推理服务 GPU。

画布组件经 mediaVideoOutput 直读落盘文件内嵌播放（同 save-video）。
"""
import json
import os
import re
import subprocess
import tempfile
import time

from flowx_client import emit, get_bytes, param, token


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", s)


def main():
    url = param("service_url").rstrip("/")
    raw = param("videos")
    xfade = param("crossfade", 0.5, float)
    prefix = sanitize(param("filename_prefix", "flowx-long"))
    out_dir = os.path.expanduser(param("output_dir", "~/flowx-output"))
    tok = token()

    ids = [s.strip() for s in raw.split(",") if s.strip()]
    if len(ids) < 2:
        raise RuntimeError(f"videos 至少需要 2 段，收到 {len(ids)}: {raw!r}")

    with tempfile.TemporaryDirectory(prefix="flowx-cc-") as td:
        srcs = []
        for i, vid in enumerate(ids):
            data = get_bytes(url, f"/videos/{vid}", tok, timeout=600)
            p = os.path.join(td, f"seg{i}.mp4")
            with open(p, "wb") as f:
                f.write(data)
            srcs.append(p)
            print(f"[concat] seg{i} video={vid} {len(data)} bytes", flush=True)

        n = len(srcs)
        # 视频：concat 滤镜（重编码，段间参数归一）
        v_in = "".join(f"[{i}:v]" for i in range(n))
        vfilter = f"{v_in}concat=n={n}:v=1:a=0[vout]"
        # 音频：acrossfade 链（a0+a1 -> a01 +a2 -> a012 ...）
        if xfade > 0:
            parts, cur = [], "0:a"
            for i in range(1, n):
                nxt = f"ax{i}"
                parts.append(f"[{cur}][{i}:a]acrossfade=d={xfade}[{nxt}]")
                cur = nxt
            afilter = ";".join(parts)
            amap = f"[{cur}]"
        else:
            a_in = "".join(f"[{i}:a]" for i in range(n))
            afilter = f"{a_in}concat=n={n}:v=0:a=1[aout]"
            amap = "[aout]"
        fc = vfilter + ";" + afilter

        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(
            out_dir, f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        cmd = (["ffmpeg", "-y"]
               + sum((["-i", p] for p in srcs), [])
               + ["-filter_complex", fc,
                  "-map", "[vout]", "-map", amap,
                  "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                  "-pix_fmt", "yuv420p",
                  "-c:a", "aac", "-b:a", "128k",
                  "-movflags", "+faststart", out])
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"ffmpeg concat failed: {r.stderr[-400:]}")

    # 时长（ffprobe）
    pr = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", out], capture_output=True, text=True, timeout=60)
    dur = 0.0
    try:
        dur = round(float(json.loads(pr.stdout)["format"]["duration"]), 2)
    except Exception:  # noqa: BLE001 - probe 失败不阻塞产出
        pass
    print(f"[concat] {n} segments -> {out} "
          f"({os.path.getsize(out)} bytes, {dur}s)", flush=True)
    emit(file_path=out, duration=dur, segments=n)


if __name__ == "__main__":
    main()
