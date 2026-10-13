#!/usr/bin/env python3
"""节流重导节点包到 Studio（运维工具）。

什么时候必须跑：
  · 改了 `_common/widget-base.js`（所有节点的控件产物都变）⇒ 全量重导；
  · 改了某节点的 `flowx.json`（版本/执行器声明/参数）或 `ui/node-widget.js`；
  · Studio 与仓库对不上时做巡检（见 §57：曾发现 29 个节点落在旧版本，
    真机跑的是**导入时的快照**，不重导等于没改）。

原理：Studio 存的是导入快照（DB + assets/nodes/<name>@<version>/），
仓库改动不会自动生效；`overwrite=true` 按节点名+版本覆盖。


Studio 限流 100 req/min/IP（burst 50）⇒ 每次间隔 1.25s（≈48/min）。
用法：python3 nodes/_tools/reimport-nodes.py [节点名…]     # 重导（节流 1.25s ≈ 48/分钟）
      python3 nodes/_tools/reimport-nodes.py --check        # 只巡检不导入（退出码 1 = 有漂移）
      （Studio 限流 100 req/min/IP；巡检同样要节流）

--check 巡检三件事（每个节点名取 Studio 里的**最新版**与仓库比）：
  1. 版本号（曾发现 29 个节点落后仓库 ⇒ 真机跑旧代码）
  2. image tag（镜像 tag 写在每个 flowx.json 里，bump 后不重导 ⇒ docker 拉旧镜像）
  3. executor.supportedTypes（曾出现 Studio 显示 local-only 而仓库已支持 docker）
"""
import json
import pathlib
import sys
import time
import urllib.error
import urllib.request

TOK = pathlib.Path.home().joinpath('.flowx-studio/auth.token').read_text().strip()
NODES = pathlib.Path(__file__).resolve().parent.parent  # nodes/
SLEEP = 1.25


def get_json(path):
    req = urllib.request.Request("http://127.0.0.1:8080" + path,
                                 headers={"Authorization": "Bearer " + TOK})
    try:
        return json.load(urllib.request.urlopen(req, timeout=120)), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:160]}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def post(path, body):
    r = urllib.request.Request('http://127.0.0.1:8080' + path,
                               data=json.dumps(body).encode(), method='POST',
                               headers={'Authorization': 'Bearer ' + TOK,
                                        'Content-Type': 'application/json'})
    try:
        return json.load(urllib.request.urlopen(r, timeout=120)), None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}: {e.read().decode()[:160]}"
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


def _verkey(v):
    parts = [int(x) for x in str(v or "0").split(".")]
    return parts + [0] * (3 - len(parts))


def check():
    """只巡检：Studio 每个名字的**最新版** vs 仓库（版本/镜像 tag/执行器声明）。"""
    repo = {}
    for fj in sorted(NODES.glob("*/flowx.json")):
        d = json.loads(fj.read_text())
        repo[d["name"]] = d
    items, page = [], 1
    while page <= 20:
        res, err = get_json(f"/api/v1/nodes?page={page}&pageSize=50")
        if err:
            print(f"巡检失败：{err}")
            return 2
        data = res.get("data") or {}
        batch = (data.get("items") if isinstance(data, dict) else data) or []
        if not batch:
            break
        items += batch
        page += 1
        time.sleep(1.0)          # 限流：100 req/min
    latest = {}
    for n in items:
        nm = n.get("name")
        if nm not in latest or _verkey(n.get("version")) > _verkey(latest[nm].get("version")):
            latest[nm] = n
    drift = []
    for nm, d in sorted(repo.items()):
        st = latest.get(nm)
        if st is None:
            drift.append(f"{nm}: Studio 里不存在（未导入？）")
            continue
        if st.get("version") != d["version"]:
            drift.append(f"{nm}: 版本 Studio={st.get('version')} 仓库={d['version']}")
        want_img = d.get("image") or (d.get("executor") or {}).get("image")
        if want_img and st.get("image") != want_img:
            drift.append(f"{nm}: image Studio={st.get('image')} 仓库={want_img}")
        sw = set((st.get("executor") or {}).get("supportedTypes") or [])
        rw = set((d.get("executor") or {}).get("supportedTypes") or [])
        if sw != rw:
            drift.append(f"{nm}: 执行器 Studio={sorted(sw)} 仓库={sorted(rw)}")
    print(f"巡检 {len(repo)} 个仓库节点 / Studio {len(items)} 个条目 / 最新版 {len(latest)} 个名字")
    if drift:
        print(f"✗ 发现 {len(drift)} 处漂移（跑 reimport-nodes.py 全量重导修复）：")
        for line in drift[:40]:
            print("   - " + line)
        if len(drift) > 40:
            print(f"   …（共 {len(drift)} 条）")
        return 1
    print("✓ 快照一致：版本 / 镜像 tag / 执行器声明 均无漂移")
    return 0


def main():
    if "--check" in sys.argv:
        return check()
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    dirs = sorted([d for d in NODES.iterdir()
                   if (d / 'flowx.json').is_file() and (not args or d.name in args)])
    print(f"待重导 {len(dirs)} 个节点（间隔 {SLEEP}s ≈ {60/SLEEP:.0f}/分钟）")
    ok = fail = 0
    t0 = time.time()
    for i, d in enumerate(dirs, 1):
        doc = json.loads((d / 'flowx.json').read_text())
        res, err = post('/api/v1/nodes/import', {
            'source_type': 'folder', 'source_path': str(d), 'overwrite': True})
        if err:
            # 409/400 之外的重试一次（隧道抖动）
            time.sleep(2.0)
            res, err = post('/api/v1/nodes/import', {
                'source_type': 'folder', 'source_path': str(d), 'overwrite': True})
        if err:
            fail += 1
            print(f"  [{i}/{len(dirs)}] FAIL {d.name}@{doc['version']}: {err}", flush=True)
        else:
            ok += 1
            if i % 10 == 0 or i == len(dirs):
                print(f"  [{i}/{len(dirs)}] ok={ok} fail={fail} "
                      f"({time.time()-t0:.0f}s)", flush=True)
        if i < len(dirs):
            time.sleep(SLEEP)
    print(f"完成：ok={ok} fail={fail} 用时 {time.time()-t0:.0f}s")
    return 1 if fail else 0


if __name__ == '__main__':
    sys.exit(main())
