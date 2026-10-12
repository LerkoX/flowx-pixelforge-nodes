#!/usr/bin/env python3
"""index-e2e-out.py：给 `~/flowx-e2e-out/` 的验收产物建索引（**只读，不动任何文件**）。

为什么需要：B1–B5 五批验收把几百个 PNG 平铺在同一个目录里（复核脚本都按 `<前缀>_<时间戳>.png`
取"最新一份"，所以**不能搬走**）。本工具不改布局，只生成 `INDEX.md` + 打印摘要：

  - 按前缀分组：份数 / 最新一份（名、大小、sha256 前 12 位）/ **不同 sha 的个数**
    （≥2 说明同参数多次运行产物不一致 ⇒ 确定性判据要看这个）
  - 按批次归类（B1–B5 的产物前缀映射，见 BATCHES）
  - 磁盘占用与文件总数

用法：python3 nodes/_tools/index-e2e-out.py [--dir ~/flowx-e2e-out] [--quiet]
"""
import argparse
import collections
import hashlib
import pathlib
import re
import sys

STAMP = re.compile(r"^(?P<prefix>.+?)_\d{8}_\d{6}(?P<ext>\.[A-Za-z0-9]+)$")

# 前缀 → 批次（人工映射；新批次在这里补一行即可）
BATCHES = [
    ("B1", ("filter_", "blur_", "sharpen_", "curves_", "hsl_", "noise_",
            "channel_", "outpaint_", "pre_")),
    ("B2", ("matting_", "matte_", "face_", "facerestore_", "defringe_")),
    ("B3", ("seg_", "segd_", "tiled_", "tiledd_")),
    ("B4", ("col_", "colin_", "ecom_", "iid", "icl_", "icld_")),
    ("B5", ("batch_",)),
]


def batch_of(prefix):
    for name, pres in BATCHES:
        if any(prefix.startswith(p) for p in pres):
            return name
    return "其他"


def sha12(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(pathlib.Path.home() / "flowx-e2e-out"))
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    root = pathlib.Path(args.dir).expanduser()
    if not root.is_dir():
        print(f"✗ 目录不存在 {root}", file=sys.stderr)
        return 1

    groups = collections.defaultdict(list)
    others = []
    for f in sorted(root.iterdir()):
        if f.is_dir():
            continue
        m = STAMP.match(f.name)
        if m:
            groups[m.group("prefix")].append(f)
        else:
            others.append(f)

    total_bytes = sum(f.stat().st_size for f in root.iterdir() if f.is_file())
    rows = []
    for prefix, files in sorted(groups.items()):
        files.sort(key=lambda p: p.stat().st_mtime)
        latest = files[-1]
        shas = {sha12(p) for p in files}
        rows.append({"prefix": prefix, "n": len(files), "latest": latest.name,
                     "kb": latest.stat().st_size // 1024, "sha12": sha12(latest),
                     "distinct_sha": len(shas), "batch": batch_of(prefix)})

    lines = ["# flowx-e2e-out 产物索引（由 nodes/_tools/index-e2e-out.py 生成，只读）", "",
             f"- 目录：`{root}`", f"- 文件总数：{sum(len(v) for v in groups.values()) + len(others)}"
             f"（前缀组 {len(groups)} 个 + 未匹配 {len(others)} 个）",
             f"- 占用：{total_bytes / 1024 ** 2:.1f} MB", "",
             "> 复核脚本按 `<前缀>_<时间戳>.png` 取**最新一份**，所以本目录**不搬家**；",
             "> `不同sha` ≥2 表示同前缀多次运行的产物不完全相同（扩散类正常，确定性类应看此项）。",
             "", "| 批次 | 前缀 | 份数 | 最新一份 | 大小 | sha256[:12] | 不同sha |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in sorted(rows, key=lambda r: (r["batch"], r["prefix"])):
        lines.append(f"| {r['batch']} | `{r['prefix']}` | {r['n']} | `{r['latest']}` | "
                     f"{r['kb']}KB | `{r['sha12']}` | {r['distinct_sha']} |")
    if others:
        lines += ["", "## 未匹配命名（不参与上面分组）", ""]
        lines += [f"- `{f.name}`（{f.stat().st_size // 1024}KB）" for f in others]
    (root / "INDEX.md").write_text("\n".join(lines) + "\n")

    if not args.quiet:
        per = collections.Counter(r["batch"] for r in rows)
        print(f"产物索引：{root}")
        print(f"  前缀组 {len(groups)} 个 / 文件 {sum(len(v) for v in groups.values())} 个 / "
              f"占用 {total_bytes / 1024 ** 2:.1f}MB")
        print("  按批次：" + "，".join(f"{k} {v} 组" for k, v in sorted(per.items())))
        multi = [r for r in rows if r["distinct_sha"] > 1]
        if multi:
            print(f"  多次运行 sha 不同（{len(multi)} 组，扩散类属预期）："
                  + "，".join(f"{r['prefix']}({r['n']}份/{r['distinct_sha']}种)"
                             for r in sorted(multi, key=lambda x: -x["n"])[:8]))
        print(f"  已写 {root / 'INDEX.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
