"""FlowX 节点共享 HTTP 客户端（纯标准库，零依赖）。
每个节点包通过 flowx.json 的 files 字段携带本文件。"""
import json
import os
import urllib.error
import urllib.request


def param(name, default=None, cast=str):
    """读取节点参数：优先裸大写环境变量，兼容 FLOWX_PARAM_* 前缀。"""
    raw = os.environ.get(name.upper()) or os.environ.get("FLOWX_PARAM_" + name.upper())
    if raw is None or raw == "":
        if default is None:
            raise RuntimeError(f"missing required parameter: {name}")
        return default
    try:
        return cast(raw)
    except (ValueError, TypeError):
        raise RuntimeError(f"parameter {name}={raw!r} cannot be cast to {cast.__name__}")


def token():
    return os.environ.get("SERVICE_TOKEN") or os.environ.get("FLOWX_PARAM_SERVICE_TOKEN") or ""


def _req(method, base, path, payload=None, tok=None, timeout=1800,
         content_type="application/json"):
    """payload 为 bytes（已编码）；GET/无 body 时传 None。"""
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, data=payload, method=method)
    if payload is not None:
        req.add_header("Content-Type", content_type)
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        raise RuntimeError(f"{method} {url} -> HTTP {e.code}: {body}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"{method} {url} failed: {e.reason}")


def post_json(base, path, payload, tok=None, timeout=1800):
    data = json.dumps(payload).encode()
    return json.loads(_req("POST", base, path, data, tok, timeout))


def post_bytes(base, path, data, tok=None, timeout=300,
               content_type="application/octet-stream"):
    """上传原始字节（如图片文件），返回解析后的 JSON 响应。"""
    return json.loads(_req("POST", base, path, data, tok, timeout, content_type))


def get_json(base, path, tok=None, timeout=60):
    return json.loads(_req("GET", base, path, None, tok, timeout))


def get_bytes(base, path, tok=None, timeout=300):
    return _req("GET", base, path, None, tok, timeout)


def call_op(base, name, inputs, tok=None, timeout=1800):
    """调能力运行时 POST /op，返回展平的 {端口: id或值}。"""
    resp = post_json(base, "/op", {"name": name, "inputs": inputs}, tok, timeout)
    return {k: m.get("id", m.get("value"))
            for k, m in resp.get("outputs", {}).items()}


def ref(oid):
    """构造对象端口引用。"""
    return {"$id": oid}


def emit(**fields):
    """以 flowx-yaml 代码块输出结果字段，供 FlowX 提取为 Metadata。"""
    print("```flowx-yaml")
    for k, v in fields.items():
        print(f'{k}: "{v}"')
    print("```")
