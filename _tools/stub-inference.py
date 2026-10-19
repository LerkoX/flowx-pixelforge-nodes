#!/usr/bin/env python3
"""stub-inference.py：推理服务**离线桩**，用于"节点改走任务式"的确定性 e2e 门禁。

用途（先例：dev-log §51.3「stub e2e」）：GPU 主机离线/不想烧显卡时间时，用本桩
替代推理服务，真跑 Studio 流水线，断言**调用协议**而不是算得快不快——
  · 是否走 `/jobs` 提交 + 轮询（任务式）而不是同步 `/op`；
  · 提交载荷的 `name`/`inputs` 是否与算子契约一致；
  · 输出是否按端口名正确回填进节点 metadata。

协议（按 flowx_client 真实形状实现）：
  GET  /ops                → {"ops":[{"name":<基名>,"plugin_hash":null}]}（核心内置 ⇒ ensure_plugin 直接命中、不传插件）
  POST /jobs               → {"job_id": "stub-<n>"}（记录载荷）
  GET  /jobs/<id>          → 前 --pending 次轮询 running，之后 done + result.outputs
  POST /op                 → 记录并 200（**出现即说明还有节点在同步调用**）
  POST /interrupt          → 记录取消请求（断言"节点被终止时是否定向取消远程任务"）
  GET  /_report            → 桩的记账 JSON，供断言用
  GET  /health             → {"status":"ok"}

用法：
  python3 nodes/_tools/stub-inference.py --port 8199 --pending 1 &
  # 临时工作流把 Param.service_url 指到 http://127.0.0.1:8199，跑完后：
  curl -s http://127.0.0.1:8199/_report | python3 -m json.tool
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# 算子 → 输出端口（id 形式的假对象；值形式的按字符串返回）
OP_OUTPUTS = {
    "flux.empty_latent": {"latent": "stublatent"},
    "flux.vae_encode": {"latent": "stublatent"},
    "flux.vae_decode": {"image": "stubimage"},
    "mesh.triposr": {"mesh": "stubmesh", "num_vertices": "1024",
                     "num_faces": "2048", "format": "glb"},
}
PENDING = 1

STATE = {"jobs": {}, "submitted": [], "sync_ops": [], "interrupts": [],
         "polls": 0}
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # 静音（桩输出只走 /_report）
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path.startswith("/ops"):
            with LOCK:
                names = sorted({s["name"] for s in STATE["submitted"]}
                               | set(OP_OUTPUTS))
            return self._send(200, {"ops": [{"name": n, "plugin_hash": None}
                                            for n in names]})
        if self.path.startswith("/jobs/"):
            jid = self.path.rsplit("/", 1)[-1]
            with LOCK:
                STATE["polls"] += 1
                rec = STATE["jobs"].get(jid)
                if rec is None:
                    return self._send(404, {"message": "no such job"})
                rec["polls"] += 1
                if rec["polls"] <= PENDING:
                    return self._send(200, {"status": "running",
                                            "progress": {"current": 0, "total": 1}})
                outs = {}
                for k, v in OP_OUTPUTS.get(rec["name"], {}).items():
                    oid = f"{v}{len(STATE['jobs']):04d}"
                    outs[k] = {"id": oid, "value": oid}
                return self._send(200, {"status": "done",
                                        "progress": {"current": 1, "total": 1},
                                        "result": {"outputs": outs}})
        if self.path.startswith("/_report"):
            with LOCK:
                return self._send(200, STATE)
        if self.path.startswith("/health"):
            return self._send(200, {"status": "ok", "cuda_available": False})
        return self._send(404, {"message": "page not found"})

    def do_POST(self):
        body = self._body()
        if self.path.startswith("/jobs"):
            with LOCK:
                jid = f"stub-{len(STATE['jobs']) + 1}"
                STATE["jobs"][jid] = {"name": body.get("name"), "polls": 0,
                                      "inputs": body.get("inputs")}
                STATE["submitted"].append({"job_id": jid, "name": body.get("name"),
                                           "inputs": body.get("inputs")})
            return self._send(200, {"job_id": jid})
        if self.path.startswith("/op"):
            with LOCK:
                STATE["sync_ops"].append(body)
            return self._send(200, {"outputs": {}})
        if self.path.startswith("/interrupt"):
            with LOCK:
                STATE["interrupts"].append(body)
            return self._send(200, {"interrupted": [body.get("job_id")]})
        if self.path.startswith("/admin/plugins"):
            return self._send(404, {"message": "stub 无插件上传（/ops 已视为内置）"})
        return self._send(404, {"message": "page not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--pending", type=int, default=1,
                    help="前 N 次轮询返回 running（>0 才真正考验轮询路径）")
    args = ap.parse_args()
    global PENDING
    PENDING = args.pending
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"stub-inference 监听 127.0.0.1:{args.port}（pending={PENDING}）", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
