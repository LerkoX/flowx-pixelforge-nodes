#!/usr/bin/env python3
r"""校验并修复工作流里的 nodeRef：任何 name@version 必须真实存在于 Studio 节点表。

教训（dev-log §60.11）：用 `text.replace("detail-refine@2.0.0", ...)` 这类**子串替换**
升级版本号会误伤 `flux-detail-refine@2.0.0` / `sd3-empty-latent@1.2.0` 这类同后缀节点名。
正确做法：正则带前边界 `(?<![\w-])name@ver`，或像本脚本这样按节点表校验后再修。
"""
import json, pathlib, re, sys, time, urllib.request

TOK = pathlib.Path.home().joinpath('.flowx-studio/auth.token').read_text().strip()


def req(method, path, body=None):
    r = urllib.request.Request('http://127.0.0.1:8080' + path,
        data=json.dumps(body).encode() if body is not None else None, method=method,
        headers={'Authorization': 'Bearer ' + TOK, 'Content-Type': 'application/json'})
    for _ in range(4):
        try:
            return json.load(urllib.request.urlopen(r, timeout=60))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15); continue
            return {'_err': f"{e.code} {e.read().decode()[:120]}"}
        except Exception:
            time.sleep(5)
    return {'_err': 'retry exhausted'}


def verkey(v):
    p = [int(x) for x in str(v).split('.')]
    return p + [0] * (3 - len(p))


def main():
    fix = "--check" not in sys.argv   # 默认校验+修复；--check 只报不改
    nodes = {}
    for page in range(1, 10):
        d = req('GET', f'/api/v1/nodes?page={page}&pageSize=50')['data']
        it = (d.get('items') if isinstance(d, dict) else d) or []
        if not it:
            break
        for n in it:
            nodes.setdefault(n['name'], []).append(n['version'])
        time.sleep(0.8)
    wfs = {}
    for page in range(1, 6):
        d = req('GET', f'/api/v1/workflows?page={page}&pageSize=50')['data']
        it = (d.get('items') if isinstance(d, dict) else d) or []
        if not it:
            break
        for w in it:
            wfs[w['id']] = w
        time.sleep(0.6)

    pat = re.compile(r'nodeRef:\s*([\w.\-]+)@([\d.]+)')
    fixed_total = 0
    for wid, w in sorted(wfs.items()):
        y = w['yamlConfig']
        bad = [(nm, v) for nm, v in pat.findall(y) if v not in nodes.get(nm, [])]
        if not bad:
            continue
        for nm, v in bad:
            avail = sorted(nodes.get(nm, []), key=verkey)
            if not avail:
                print(f"  wf{wid}: {nm}@{v} —— Studio 无此节点名，跳过（需人工）")
                continue
            target = avail[-1]
            y = pat.sub(lambda m: f"nodeRef: {m.group(1)}@{target}"
                        if (m.group(1) == nm and m.group(2) == v) else m.group(0), y)
            print(f"  wf{wid}: {nm}@{v} -> @{target}（Studio 可选 {avail}）")
            fixed_total += 1
        if not fix:
            print(f"  wf{wid}: 只报不改（--check）")
            continue
        res = req('PUT', f'/api/v1/workflows/{wid}', {'name': w['name'], 'yamlConfig': y})
        if '_err' in res:
            print(f"    写入失败: {res['_err']}")
        time.sleep(0.6)
    print(f"共{('修复' if fix else '发现')} {fixed_total} 处非法 nodeRef")
    return 1 if (fixed_total and not fix) else 0


if __name__ == '__main__':
    sys.exit(main())
