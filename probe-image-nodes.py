#!/usr/bin/env python3
"""镜像内节点启动探针：确认「代码真的在镜像里且能起来」。

背景：cpolar 隧道上 `docker run`（attach 模式）的输出会被静默截断，看不到任何日志，
所以本脚本用 detached 启动 + `docker logs` 取输出（与 flowx docker 执行器修复后同一思路）。

判定：
  OK     —— 节点跑到了自己的逻辑并因"缺参数/连不上服务"退出（预期）
  BROKEN —— 环境问题：找不到文件/模块、语法错误、权限错误等

用法：
  DOCKER_HOST=tcp://<daemon> python3 nodes/probe-image-nodes.py
  python3 nodes/probe-image-nodes.py --tag lerkobba/flowx-pixelforge-nodes:v1.6.0
"""
from __future__ import annotations

import argparse
import concurrent.futures
import pathlib
import re
import subprocess
import sys

TUNNEL_PATTERNS = ["error during connect", "connection reset by peer", "i/o timeout", "EOF"]
BROKEN_PATTERNS = [
    ("No such file or directory", "文件/目录缺失"),
    ("can't open file", "入口文件缺失"),
    ("ModuleNotFoundError", "依赖缺失"),
    ("ImportError", "导入失败"),
    ("SyntaxError", "语法错误"),
    ("Permission denied", "权限问题"),
    ("not found in PATH", "可执行文件缺失"),
]
OK_PATTERNS = [
    ("missing required parameter", "缺参数（预期）"),
    ("RuntimeError", "节点自身报错（预期）"),
    ("URLError", "连不上服务（预期）"),
    ("Connection refused", "连不上服务（预期）"),
    ("HTTPError", "服务返回错误（预期）"),
    ("TimeoutError", "服务超时（预期）"),
    ("flowx", "flowx 协议输出"),
]


def in_image_dirs(root: pathlib.Path) -> list[str]:
    out = []
    for line in (root / "Dockerfile").read_text().splitlines():
        m = re.match(r"^COPY\s+([\w\-.]+)/\*\.py\s+\S+/?\s*$", line.strip())
        if m:
            out.append(m.group(1))
    return out


def classify(logs: str) -> tuple[str, str]:
    # 隧道抖动（cpolar）：与节点无关，由调用方重试，不计入镜像问题
    if any(p in logs for p in TUNNEL_PATTERNS):
        return "TUNNEL", "隧道抖动（重试）"
    for pat, why in BROKEN_PATTERNS:
        if pat in logs:
            return "BROKEN", why
    for pat, why in OK_PATTERNS:
        if pat in logs:
            return "OK", why
    return ("UNKNOWN", "无匹配信号")


def probe(tag: str, name: str, wait: float, attempts: int = 2) -> tuple[str, str, str]:
    for attempt in range(1, attempts + 1):
        status, why, tail = _probe_once(tag, name, wait, attempt)
        if status != "TUNNEL":
            return status, why, tail
    return "TUNNEL", f"隧道连续 {attempts} 次抖动", tail


def _probe_once(tag: str, name: str, wait: float, attempt: int) -> tuple[str, str, str]:
    cname = f"flowx-probe-{name}"
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
    run = subprocess.run(
        ["docker", "run", "-d", "--name", cname, "-e", "SERVICE_URL=http://127.0.0.1:9", tag,
         "sh", "-c", f"cd /opt/flowx-nodes/{name} && timeout 20 python main.py; echo PROBE_EXIT=$?"],
        capture_output=True, text=True, timeout=180,
    )
    if run.returncode != 0:
        return "TUNNEL", "docker run 失败（隧道）", run.stderr.strip()[:200]
    import time
    time.sleep(wait)
    logs = subprocess.run(["docker", "logs", cname], capture_output=True, text=True, timeout=120)
    out = (logs.stdout or "") + (logs.stderr or "")
    # 清理失败（隧道抖动）不影响判定，但要避免容器堆积
    subprocess.run(["docker", "rm", "-f", cname], capture_output=True)
    status, why = classify(out)
    tail = " / ".join(line.strip() for line in out.strip().splitlines()[-2:])[:220]
    return status, why, tail


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--wait", type=float, default=2.0)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    root = pathlib.Path(__file__).parent
    tag = args.tag or ("lerkobba/flowx-pixelforge-nodes:v" + (root / "BUNDLE_VERSION").read_text().strip())
    dirs = in_image_dirs(root)
    print(f"探针镜像 {tag}，{len(dirs)} 个节点目录（detached + docker logs）\n")

    results: dict[str, tuple[str, str, str]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(probe, tag, d, args.wait): d for d in dirs}
        for fut in concurrent.futures.as_completed(futs):
            d = futs[fut]
            try:
                results[d] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[d] = ("BROKEN", "探针异常", str(exc)[:200])

    broken = tunnel = 0
    for d in sorted(dirs):
        status, why, tail = results[d]
        if status == "BROKEN":
            broken += 1
        elif status == "TUNNEL":
            tunnel += 1
        mark = {"BROKEN": "✗", "OK": "✓", "TUNNEL": "~"}.get(status, "?")
        print(f"  {mark} {d:<24} {why:<16} {tail}")

    print(f"\n正常 {sum(1 for r in results.values() if r[0] == 'OK')}/{len(dirs)}；"
          f"BROKEN {broken}；隧道未确定 {tunnel}")
    if tunnel:
        print("（隧道未确定的节点请重跑本脚本补齐；BROKEN 才是镜像问题）")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main())
