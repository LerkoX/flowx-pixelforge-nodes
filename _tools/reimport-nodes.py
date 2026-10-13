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
用法：python3 nodes/_tools/reimport-nodes.py [节点名…]
      （间隔 1.25s ≈ 48/分钟，避开 Studio 的 100 req/min/IP 限流）
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


def main():
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
