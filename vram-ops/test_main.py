#!/usr/bin/env python3
"""vram-ops 本机单测：真起 HTTP 桩，跑 main()，断言请求形状与输出字段。

不需要 GPU/真实推理服务；覆盖六个动作 + 非法 action + 服务端缺端点时的报错。
运行：python3 nodes/vram-ops/test_main.py
"""
import importlib.util
import io
import json
import os
import pathlib
import sys
import threading
from contextlib import redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

FAILED = []
REQ = []          # 桩收到的 (method, path, body, headers)


def check(label, cond, extra=""):
    print(("PASS " if cond else "FAIL ") + label + (f"  [{extra}]" if extra and not cond else ""))
    if not cond:
        FAILED.append(label)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record(self, method):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            body = json.loads(raw) if raw else None
        except ValueError:
            body = raw.decode(errors="replace")
        REQ.append((method, self.path, body, dict(self.headers)))

    def do_GET(self):
        self._record("GET")
        if self.path.startswith("/health"):
            return self._send(200, {"status": "ok", "gpu_name": "GTX 1080",
                                    "vram_free_mb": 6956,
                                    "resident_models": ["flux-vae:vae", "flux-clip:t5"]})
        if self.path.startswith("/leases"):
            return self._send(200, {"leases": [{"lease": "724", "keys": ["flux-vae:vae"],
                                                "idle_seconds": 1296.3}],
                                    "lease_ttl_seconds": 1800.0,
                                    "lease_wait_timeout_seconds": 300.0})
        if self.path.startswith("/model/policy"):
            return self._send(200, {"max_resident_models": 2, "vram_reserve_mb": 1024,
                                    "lease_ttl_seconds": 1800.0, "offline": None,
                                    "quantization": "none", "offload_mode": "none"})
        return self._send(404, {"detail": "page not found"})

    def do_POST(self):
        self._record("POST")
        if self.path.startswith("/model/unload"):
            return self._send(200, {"unloaded": "flux-vae:vae", "resident": "flux-clip:t5"})
        if self.path.startswith("/interrupt"):
            return self._send(200, {"interrupted": ["stub-1"]})
        if self.path.startswith("/model/policy"):
            return self._send(200, {"applied": {"vram_reserve_mb": 768},
                                    "policy": {"max_resident_models": 2, "vram_reserve_mb": 768}})
        return self._send(404, {"detail": "page not found"})


def load_mod():
    path = pathlib.Path(__file__).with_name("main.py")
    spec = importlib.util.spec_from_file_location("vram_ops_main", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run(mod, **env):
    base = {"SERVICE_URL": "http://127.0.0.1:8211", "SERVICE_TOKEN": ""}
    base.update({k: str(v) for k, v in env.items()})
    old = {k: os.environ.get(k) for k in base}
    old.update({k: os.environ.get("FLOWX_PARAM_" + k) for k in base})
    os.environ.update(base)
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            mod.main()
    except Exception as e:  # noqa: BLE001
        return None, buf.getvalue(), f"{type(e).__name__}: {e}"
    finally:
        for k, v in old.items():
            if k.startswith("FLOWX_PARAM_"):
                continue
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    out = {}
    lines = buf.getvalue().splitlines()
    if "```flowx-yaml" in lines:
        i = lines.index("```flowx-yaml")
        for line in lines[i + 1:]:
            if line.startswith("```"):
                break
            if ": " in line:
                k, v = line.split(": ", 1)
                out[k] = v.strip().strip('"')
    return out, buf.getvalue(), None


def main():
    srv = ThreadingHTTPServer(("127.0.0.1", 8211), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    mod = load_mod()

    # ① status
    REQ.clear()
    out, _, err = run(mod, ACTION="status")
    check("status：无异常", err is None, err or "")
    check("status：GET /health", REQ and REQ[-1][1].startswith("/health"))
    check("status：resident_count=2", out and out.get("resident_count") == "2", str(out))
    check("status：vram_free_mb=6956", out and out.get("vram_free_mb") == "6956", str(out))
    check("status：detail 人可读", out and "GTX 1080" in out.get("detail", ""), str(out))

    # ② leases
    REQ.clear()
    out, _, err = run(mod, ACTION="leases")
    check("leases：GET /leases", REQ and REQ[-1][1].startswith("/leases"))
    check("leases：lease_count=1", out and out.get("lease_count") == "1", str(out))
    check("leases：detail 含租约号与持有项",
          out and "724" in out.get("detail", "") and "flux-vae" in out.get("detail", ""), str(out))
    check("leases：detail 含 TTL 提示", out and "ttl=1800" in out.get("detail", ""), str(out))

    # ③ unload（带租约头）
    REQ.clear()
    out, _, err = run(mod, ACTION="unload", TARGET="vae", LEASE="724")
    m, path, body, headers = REQ[-1]
    check("unload：POST /model/unload + body.target",
          m == "POST" and path.startswith("/model/unload") and body == {"target": "vae"}, str(body))
    # HTTP 头大小写不敏感（urllib 会规范成 X-Flowx-Lease），按小写比较
    hdr_lower = {k.lower(): v for k, v in headers.items()}
    check("unload：透传 X-FlowX-Lease 头", hdr_lower.get("x-flowx-lease") == "724", str(headers))
    check("unload：detail 含 unloaded", out and "unloaded=flux-vae" in out.get("detail", ""), str(out))

    # ④ interrupt（指定 job / 全部）
    REQ.clear()
    out, _, err = run(mod, ACTION="interrupt", JOB_ID="stub-1")
    check("interrupt：body 带 job_id", REQ[-1][2] == {"job_id": "stub-1"}, str(REQ[-1][2]))
    REQ.clear()
    out, _, err = run(mod, ACTION="interrupt")
    check("interrupt：job_id 空 ⇒ body 为空对象（全部）", REQ[-1][2] == {}, str(REQ[-1][2]))

    # ⑤ policy_get / policy_set
    REQ.clear()
    out, _, err = run(mod, ACTION="policy_get")
    check("policy_get：GET /model/policy", REQ and REQ[-1][1].startswith("/model/policy"))
    check("policy_get：policy 串为 k=v 形式（无花括号）",
          out and "max_resident_models=2" in out.get("policy", "")
          and "{" not in out.get("policy", ""), str(out))
    REQ.clear()
    out, _, err = run(mod, ACTION="policy_set", POLICY_JSON='{"vram_reserve_mb": 768}')
    check("policy_set：body 即 patch", REQ[-1][2] == {"vram_reserve_mb": 768}, str(REQ[-1][2]))
    check("policy_set：detail 含 applied", out and "applied=vram_reserve_mb=768" in out.get("detail", ""), str(out))

    # ⑥ 非法 action / 非法 JSON：报错清晰（不静默）
    _, _, err = run(mod, ACTION="nonsense")
    check("非法 action 报错且列出可选值", err and "action 非法" in err, err or "")
    _, _, err = run(mod, ACTION="policy_set", POLICY_JSON="{not json")
    check("policy_json 非法 JSON 报错", err and "不是合法 JSON" in err, err or "")

    # ⑦ 输出值不得含 # 或代码围栏（gate N：Studio codec 会截断）
    out, raw, err = run(mod, ACTION="leases")
    vals = " ".join(str(v) for v in (out or {}).values())
    check("输出值不含 '#'", "#" not in vals, vals)
    check("输出值不含代码围栏", "```" not in vals, vals)

    srv.shutdown()
    print()
    print(f"PASSED = {len(FAILED) == 0}  FAILED = {len(FAILED)}")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
