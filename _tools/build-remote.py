#!/usr/bin/env python3
"""经 Docker Engine API 构建/推送节点共享镜像（无 docker CLI 时的等效工具）。

**为什么需要**：本仓的主力开发环境是 Termux（Android），装不了 docker CLI；
而 GPU 机器的 daemon 经 ssh tunnel / cpolar 隧道可达。此时 `build-image.sh`
（走 CLI）跑不了，本脚本用 HTTP API 完成同一件事。

行为与 `docker build` 一致的关键点：
  · 上下文严格按 `.dockerignore` 裁剪（`*/ui/`、`*/mock.py`、`_tools/`、`.git/`、
    `_common/`、文档… 根本不进上下文，也不进镜像层）；
  · 默认 gzip（`Content-Encoding: gzip`，24MB → 14MB）。cpolar 隧道对大 POST
    不稳定（`RemoteDisconnected` / `Connection reset by peer`），压缩显著提升成功率；
  · 失败自动重试（`--retries`，默认 3），指数退避。

用法（在仓库根或任意目录）：
  # 回家模式（局域网直连，13ms）
  python3 _tools/build-remote.py --host tcp://192.168.5.60:2375

  # 出门模式（cpolar 隧道；地址每次不同，见 flowx-mode.sh）
  python3 _tools/build-remote.py --host tcp://7.tcp.vip.cpolar.cn:13344

  # 只看会传什么（不打 daemon）
  python3 _tools/build-remote.py --host tcp://... --dry-run

  默认 tag = lerkobba/flowx-pixelforge-nodes:v<BUNDLE_VERSION>；可用 --tag 覆盖。
  host 也可来自环境变量 FLOWX_DOCKER_HOST / DOCKER_HOST。

发布顺序（同 README「顺序纪律」）：
  bump BUNDLE_VERSION → sync-flowx-json.py → 本脚本（或 build-image.sh）
  → reimport-nodes.py 全量重导（tag 写在每个 flowx.json 里，漏了会拉旧镜像）

注意：API 构建落在**远端 daemon 本地**，不推 Docker Hub。要给别的机器用，
另跑 `docker push`（需要 registry 凭证）。
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import pathlib
import re
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_REPO = "lerkobba/flowx-pixelforge-nodes"
COPY_RE = re.compile(r'^\s*COPY\s+(?:--\S+\s+)*(.+)$')
WHOLE_SRC = (".", "./")


# ---------------------------------------------------------------- 上下文收集

def parse_copy_sources(dockerfile: pathlib.Path) -> list[str]:
    """Dockerfile 里所有 COPY 的源（相对上下文根）；忽略 --from 与目标。"""
    sources = []
    for raw in dockerfile.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = COPY_RE.match(line)
        if not m:
            continue
        body = m.group(1)
        if body.startswith("["):
            try:
                parts = json.loads(body)
            except json.JSONDecodeError:
                continue
            srcs = parts[:-1]
        else:
            srcs = body.split()[:-1]
        sources.extend(s for s in srcs if not s.startswith("--"))
    return sources


def load_dockerignore(context: pathlib.Path) -> list[str]:
    f = context / ".dockerignore"
    if not f.is_file():
        return []
    return [s.strip() for s in f.read_text().splitlines()
            if s.strip() and not s.strip().startswith("#")]


def is_ignored(rel: str, pats: list[str]) -> bool:
    """按 .dockerignore 语义判断（覆盖本仓用到的几种形态）。"""
    parts = rel.split("/")
    for p in pats:
        if p.endswith("/"):
            base = p[:-1]
            if base.startswith("**/"):
                if any(seg == base[3:] for seg in parts):
                    return True
            elif "/" in base:
                if rel == base or rel.startswith(base + "/") \
                        or fnmatch.fnmatch(rel, base + "/*"):
                    return True
            elif rel == base or rel.startswith(base + "/"):
                return True
        elif fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(parts[-1], p):
            return True
    return False


def collect(context: pathlib.Path, sources: list[str]) -> list[pathlib.Path]:
    """展开 COPY 源为文件清单（应用 .dockerignore + 跳过 __pycache__）。"""
    ignore = load_dockerignore(context)
    files: list[pathlib.Path] = []
    seen: set[str] = set()

    def expand(p: pathlib.Path):
        if p.is_dir():
            yield from sorted(p.rglob("*"))
        elif p.is_file():
            yield p

    for src in sources:
        if src.rstrip("/") in WHOLE_SRC:
            matches = [context]                 # COPY . → 整个上下文
        elif any(ch in src for ch in "*?["):
            matches = sorted(context.glob(src))
        else:
            matches = [context / src]
        for mt in matches:
            for f in expand(mt):
                if not f.is_file() or "__pycache__" in f.parts:
                    continue
                rel = f.relative_to(context).as_posix()
                if rel in seen or is_ignored(rel, ignore):
                    continue
                seen.add(rel)
                files.append(f)
    df = context / "Dockerfile"
    if str(df) not in seen:
        files.append(df)
    return files


def build_tar(context: pathlib.Path, files: list[pathlib.Path],
              compress: bool = True) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz" if compress else "w") as tar:
        for f in files:
            ti = tar.gettarinfo(str(f), arcname=f.relative_to(context).as_posix())
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            with open(f, "rb") as fh:
                tar.addfile(ti, fh)
    return buf.getvalue()


# ---------------------------------------------------------------- Engine API

def to_base_url(host: str) -> str:
    return host.replace("tcp://", "http://").rstrip("/")


def build(host: str, blob: bytes, tag: str, dockerfile: str,
          compress: bool, no_cache: bool) -> tuple[bool, str]:
    """POST /build，流式读日志。返回 (成功?, 错误摘要)。"""
    params = {"t": tag, "dockerfile": dockerfile}
    if no_cache:
        params["nocache"] = "1"
    url = f"{to_base_url(host)}/build?" + urllib.parse.urlencode(params)
    headers = {"Content-Type": "application/x-tar"}
    if compress:
        headers["Content-Encoding"] = "gzip"
    req = urllib.request.Request(url, data=blob, method="POST", headers=headers)

    print(f"POST {url}", flush=True)
    print(f"（上下文 {len(blob) / 1e6:.1f} MB"
          f"{' gzip' if compress else ''}）", flush=True)
    t0, last = time.time(), ""
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            for raw in resp:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if "error" in ev:
                    msg = ev.get("errorDetail", {}).get("message") or ev["error"]
                    return False, str(msg)
                st = (ev.get("stream") or "").rstrip()
                if st and st != last:
                    print(st[:300], flush=True)
                    last = st
                if ev.get("aux", {}).get("ID"):
                    print(f"image id = {ev['aux']['ID']}", flush=True)
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:1500]}"
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        return False, f"{type(e).__name__}: {e}"
    print(f"构建耗时 {time.time() - t0:.0f}s", flush=True)
    return True, ""


def inspect(host: str, tag: str) -> dict | None:
    try:
        with urllib.request.urlopen(f"{to_base_url(host)}/images/{tag}/json",
                                    timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError:
        return None


# ---------------------------------------------------------------- main

def main() -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    bundle = (root / "BUNDLE_VERSION").read_text().strip() \
        if (root / "BUNDLE_VERSION").is_file() else "0"

    ap = argparse.ArgumentParser(description="经 Docker Engine API 构建节点镜像")
    ap.add_argument("--host", default=os.environ.get("FLOWX_DOCKER_HOST")
                    or os.environ.get("DOCKER_HOST"),
                    help="tcp://host:port 或 http://host:port"
                         "（默认读 FLOWX_DOCKER_HOST / DOCKER_HOST）")
    ap.add_argument("--context", default=str(root))
    ap.add_argument("--tag", default=f"{DEFAULT_REPO}:v{bundle}")
    ap.add_argument("--dockerfile", default="Dockerfile")
    ap.add_argument("--no-gzip", action="store_true",
                    help="不压缩上下文（默认 gzip）")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--retries", type=int, default=3,
                    help="网络类失败重试次数（默认 3，指数退避）")
    ap.add_argument("--dry-run", action="store_true",
                    help="只统计上下文，不连 daemon")
    args = ap.parse_args()

    context = pathlib.Path(args.context).expanduser().resolve()
    sources = parse_copy_sources(context / args.dockerfile)
    files = collect(context, sources)
    compress = not args.no_gzip
    blob = build_tar(context, files, compress=compress)
    raw_mb = sum(f.stat().st_size for f in files) / 1e6
    print(f"上下文：{len(sources)} 条 COPY → {len(files)} 文件 / {raw_mb:.1f} MB"
          f" → tar{' .gz' if compress else ''} {len(blob) / 1e6:.1f} MB | tag={args.tag}",
          flush=True)
    if args.dry_run:
        for f in files[:8]:
            print("  ", f.relative_to(context))
        print("   ...（--dry-run：未连接 daemon）")
        return 0

    if not args.host:
        print("✗ 缺 --host（或用 FLOWX_DOCKER_HOST / DOCKER_HOST 指定 daemon 地址）",
              file=sys.stderr)
        return 2

    last_err = ""
    for attempt in range(1, max(1, args.retries) + 1):
        if attempt > 1:
            wait = min(5 * 2 ** (attempt - 2), 30)
            print(f"\n[重试 {attempt}/{args.retries}] 上次失败：{last_err[:160]}；"
                  f"{wait}s 后重试", flush=True)
            time.sleep(wait)
        ok, err = build(args.host, blob, args.tag, args.dockerfile,
                        compress, args.no_cache)
        if ok:
            break
        last_err = err
    else:
        print(f"\n✗ 构建失败：{last_err}", file=sys.stderr)
        return 1

    info = inspect(args.host, args.tag)
    if not info:
        print(f"✗ 构建返回成功但镜像不落地：{args.tag}", file=sys.stderr)
        return 1
    print(f"\n✓ {args.tag} | {info.get('Id', '')[:19]} | "
          f"{(info.get('Size') or 0) / 1e6:.1f} MB | 层数 "
          f"{len(info.get('RootFS', {}).get('Layers') or [])}")
    print("  下一步：`python3 _tools/reimport-nodes.py` 全量重导（tag 写在每个 "
          "flowx.json 里）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
