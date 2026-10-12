"""批量 I/O 共享件（B5/52.2）：目录扫描 / 上传 / 下载 / 命名。

三个批量节点（`image-batch-load` / `image-batch-save` / `image-montage`）共用。
**单一事实源**：路径展开、排序、分页、`@` 信封、文件名规则只在这里定义一次，
避免 load/save 两侧对"序号从几开始""补零几位"理解不一致（gate N 桩冒烟也会跑这里）。

约束：零 numpy、零 GPU、纯标准库 + `flowx_client`（HTTP）。本机（Termux）可跑纯逻辑单测。
"""
import fnmatch
import json
import os

from flowx_client import get_bytes, post_bytes

_SORT_KEYS = {
    "name": lambda p: os.path.basename(p).lower(),
    "mtime": lambda p: os.path.getmtime(p),
    "size": lambda p: os.path.getsize(p),
}

_CTYPE = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
          ".webp": "image/webp", ".bmp": "image/bmp"}


def dumps(obj):
    """JSON 负载的信封编码：`@` + JSON（Studio 会把"像 JSON 的字符串"自动反序列化成
    map，注入节点 env 时变成 Go 的 `<map[string]interface {} Value>`，B3 真机复现）。"""
    return "@" + json.dumps(obj, ensure_ascii=False)


def loads(raw):
    """`@` 信封解码；容忍裸 JSON 与已解析的 list；空值给空列表。"""
    if raw is None or raw == "":
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    if not isinstance(raw, str):
        return [raw]
    t = raw[1:] if raw.startswith("@") else raw
    try:
        v = json.loads(t)
    except Exception:
        return [s for s in (x.strip() for x in t.split(",")) if s]
    return list(v) if isinstance(v, (list, tuple)) else [v]


def pbool(raw):
    """布尔参数解析（env 注入是字符串，`bool("false") == True` 是真陷阱）。"""
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def expand_dir(dirpath):
    """`~` 展开 + 绝对化（不存在时给可操作报错）。"""
    path = os.path.abspath(os.path.expanduser(str(dirpath or "").strip()))
    if not os.path.isdir(path):
        raise FileNotFoundError(f"image-batch: 目录不存在 {path}")
    return path


def collect(dirpath, pattern="*.png", sort="name", reverse="false", limit=0,
            offset=0, recursive="false"):
    """扫描目录 → (文件绝对路径列表, 目录绝对路径)。

    pattern 多组用 `;` 分隔（如 `*.png;*.jpg`）；只按**文件名**匹配（不匹配目录名）。
    sort ∈ name|mtime|size；limit=0 表示全部；offset 先跳过再取 limit（分页）。
    """
    path = expand_dir(dirpath)
    pats = [p.strip().lower() for p in str(pattern or "*").split(";") if p.strip()] or ["*"]
    files = []
    if pbool(recursive):
        for root, _dirs, names in os.walk(path):
            for n in names:
                files.append(os.path.join(root, n))
    else:
        for n in os.listdir(path):
            full = os.path.join(path, n)
            if os.path.isfile(full):
                files.append(full)
    key = _SORT_KEYS.get(str(sort or "name").strip().lower())
    if key is None:
        raise ValueError(f"image-batch: sort 非法 {sort!r}（可选 {sorted(_SORT_KEYS)}）")
    files.sort(key=key)
    if pbool(reverse):
        files.reverse()
    files = [f for f in files
             if any(fnmatch.fnmatch(os.path.basename(f).lower(), p) for p in pats)]
    off = max(0, int(offset or 0))
    lim = int(limit or 0)
    files = files[off:] if lim <= 0 else files[off:off + lim]
    return files, path


def content_type(path):
    return _CTYPE.get(os.path.splitext(path)[1].lower(), "image/png")


def load_batch(url, files, tok=None):
    """逐文件上传到对象仓库 → (ID 列表 JSON, 文件名列表 JSON)。"""
    ids, names = [], []
    for f in files:
        with open(f, "rb") as fh:
            data = fh.read()
        r = post_bytes(url, "/images", data, tok, timeout=300,
                       content_type=content_type(f))
        oid = r.get("id") or r.get("$id") or (r.get("outputs") or {}).get("id")
        if not oid:
            raise RuntimeError(f"image-batch: 上传 {os.path.basename(f)} 失败：{r}")
        ids.append(oid)
        names.append(os.path.basename(f))
    return dumps(ids), dumps(names)


def get_pimg_bytes(url, oid, tok=None):
    """对象 ID → 原始字节（拼图节点取图用；与 save_batch 同一取数口径）。"""
    return get_bytes(url, f"/images/{oid}", tok, timeout=300)


def load_ids(raw):
    """解析 `images` 参数（`@`/裸 JSON/逗号分隔都认）。"""
    out = []
    for v in loads(raw):
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def save_batch(url, ids, output_dir, prefix="batch", start_index=1, zero_pad=3,
               ext="png", tok=None):
    """对象 ID 列表 → 落盘 `[output_dir]/<prefix>_<序号>.<ext>` → (paths, names, dir)。"""
    if not ids:
        return dumps([]), dumps([]), os.path.abspath(os.path.expanduser(str(output_dir)))
    outdir = os.path.abspath(os.path.expanduser(str(output_dir or "").strip()))
    os.makedirs(outdir, exist_ok=True)
    e = str(ext or "png").strip().lstrip(".").lower()
    if e not in ("png", "jpg", "jpeg", "webp"):
        raise ValueError(f"image-batch-save: ext 非法 {ext!r}（png/jpg/webp）")
    pad = max(0, int(zero_pad or 0))
    idx = int(start_index or 1)
    paths, names = [], []
    for i, oid in enumerate(ids):
        data = get_bytes(url, f"/images/{oid}", tok, timeout=300)
        name = (f"{prefix}_{idx + i:0{pad}d}.{e}" if pad
                else f"{prefix}_{idx + i}.{e}")
        full = os.path.join(outdir, name)
        with open(full, "wb") as fh:
            fh.write(data)
        paths.append(full)
        names.append(name)
    return dumps(paths), dumps(names), outdir
