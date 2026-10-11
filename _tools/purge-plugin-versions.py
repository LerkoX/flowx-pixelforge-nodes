#!/usr/bin/env python3
"""purge-plugin-versions.py：清掉服务端 plugins.d 里**被取代**的族版本文件（F1 配套）。

背景（dev-plan §47）：F1 之后算子身份 = 实现内容，族文件按内容寻址上传成
`<OP_FAMILY>_<OP_VERSION>.py`。好处是新旧版本并存、互不顶替；代价是 plugins.d 会随
发版累积版本文件（每个都注册着 <基名>@<hash> 这套算子）。

本工具只删「本地当前族文件的版本戳之外」的同族版本文件：
  - 判定依据是本地 `_common/server_ops/*.py` 的 OP_FAMILY / OP_VERSION（唯一事实源），
    因此绝不会误删当前版本；
  - 未版本化的历史文件（如 `preprocess_canny.py`）**不动** —— 老 bundle 节点还在用；
  - 删除会让旧版本算子在服务端注销；若某个旧版本节点随后再跑，它会自行重新上传
    （内容寻址，幂等），属于自愈行为。

用法：
  python3 nodes/_tools/purge-plugin-versions.py                 # 默认 dry-run，只列出
  python3 nodes/_tools/purge-plugin-versions.py --apply         # 真删
  python3 nodes/_tools/purge-plugin-versions.py --apply --keep-extra abc123,def456
环境变量：FLOWX_SERVICE_URL / FLOWX_TOKEN（与节点参数同口径），或用 --url/--token。
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
STAMP_RE = re.compile(r'^OP_VERSION\s*=\s*"([0-9a-f]+)"', re.M)
FAMILY_RE = re.compile(r'^OP_FAMILY\s*=\s*"([\w\.]+)"', re.M)


def local_versions():
    """{OP_FAMILY: OP_VERSION}：本地当前族版本（唯一事实源）。"""
    out = {}
    for path in sorted((ROOT / "_common" / "server_ops").glob("*.py")):
        src = path.read_text()
        fam = FAMILY_RE.search(src)
        ver = STAMP_RE.search(src)
        if fam and ver:
            out[fam.group(1)] = ver.group(1)
    return out


def req(base, path, tok, method="GET"):
    r = urllib.request.Request(base + path, method=method)
    if tok:
        r.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(r, timeout=60) as f:
            return 200, json.load(f)
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("FLOWX_SERVICE_URL", ""))
    ap.add_argument("--token", default=os.environ.get("FLOWX_TOKEN", ""))
    ap.add_argument("--apply", action="store_true", help="真删（默认只列出）")
    ap.add_argument("--keep-extra", default="",
                    help="额外保留的版本 hash（逗号分隔，如正在验证的旧版）")
    a = ap.parse_args()
    if not a.url or not a.token:
        print("✗ 需要 --url/--token（或 FLOWX_SERVICE_URL/FLOWX_TOKEN）")
        return 1
    base = a.url.rstrip("/")
    keep_extra = {h.strip() for h in a.keep_extra.split(",") if h.strip()}
    versions = local_versions()
    code, data = req(base, "/admin/plugins", a.token)
    if code != 200:
        print(f"✗ GET /admin/plugins -> {code}: {data}")
        return 1
    files = data.get("plugins", {})
    doomed, warned = [], []
    for fam, ver in sorted(versions.items()):
        pat = re.compile(rf"^{re.escape(fam)}_([0-9a-f]{{12}})\.py$")
        for fn in sorted(files):
            m = pat.match(fn)
            if not m or m.group(1) == ver or m.group(1) in keep_extra:
                continue
            doomed.append((fn, fam, m.group(1), ver, files[fn]["ops"]))
    # 名字像族版本文件、但本地已没有该族（族被删/改名）的：只警告不删
    known = set(versions)
    for fn in sorted(files):
        m = re.match(r"^([\w]+)_([0-9a-f]{12})\.py$", fn)
        if m and m.group(1) not in known:
            warned.append((fn, files[fn]["ops"]))
    print("本地族版本（保留）："
          + ", ".join(f"{k}@{v}" for k, v in sorted(versions.items())))
    if warned:
        print("\n⚠ 以下像版本化族文件、但其族在本地已不存在（保守起见不删，请人工确认）：")
        for fn, ops in warned:
            print(f"   {fn}  注册 {len(ops)} 个算子")
    if not doomed:
        print("\n✓ 没有被取代的旧版本文件（无需清理）")
        return 0
    print(f"\n将被删除的旧版本文件（{len(doomed)} 个）：")
    for fn, fam, old, cur, ops in doomed:
        print(f"   {fn}  ({fam} 旧版 {old} → 当前 {cur}，{len(ops)} 个算子)")
    if not a.apply:
        print("\n（dry-run；加 --apply 执行删除）")
        return 0
    rc = 0
    for fn, *_ in doomed:
        code, data = req(base, f"/admin/plugins/{fn}", a.token, method="DELETE")
        ok = code == 200
        rc |= 0 if ok else 1
        print(("✓ 已删除 " if ok else f"✗ 删除失败({code}) ") + fn)
    return rc


if __name__ == "__main__":
    sys.exit(main())
