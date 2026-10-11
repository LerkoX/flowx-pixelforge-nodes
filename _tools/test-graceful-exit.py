#!/usr/bin/env python3
"""test-graceful-exit.py：验证 flowx_client 的优雅退出契约。

场景：Studio 终止执行 → flowx 本地执行器先发 SIGTERM（宽限期内不退出才 SIGKILL）。
节点进程必须在收到 SIGTERM 时取消自己在推理服务上 running 的 job，
否则远程任务变孤儿继续占 GPU。

用真实子进程 + 本地假推理服务端到端验证：
  1. 子进程 submit_job（登记为进行中任务）；
  2. 父进程发 SIGTERM；
  3. 断言假服务收到 POST /interrupt 且 job_id 匹配；
  4. 断言子进程以 128+SIGTERM 退出（说明走的是信号处理器，不是被强杀）。

用法：python3 nodes/_tools/test-graceful-exit.py   （退出码 0=通过）
"""
import json
import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMMON = ROOT / "_common"

CHILD = r'''
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
import flowx_client as fc
base, tok, ready = sys.argv[2], sys.argv[3], sys.argv[4]
jid = fc.submit_job(base, {"name": "flux.mask_refine", "inputs": {}}, tok)
with open(ready, "w") as f:
    f.write(jid)
print("SUBMITTED " + jid, flush=True)
time.sleep(60)
'''


def main():
    received = []
    jobs_submitted = []

    class Handler(BaseHTTPRequestHandler):
        def _read(self):
            n = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(n).decode() if n else ""

        def do_POST(self):
            body = self._read()
            if self.path == "/jobs":
                jobs_submitted.append(body)
                payload = {"job_id": "job-0123456789abcdef"}
            elif self.path == "/interrupt":
                received.append({"body": json.loads(body or "{}"),
                                 "auth": self.headers.get("Authorization"),
                                 "at": time.time()})
                payload = {"ok": True}
            else:
                payload = {}
            data = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"

    with tempfile.TemporaryDirectory() as td:
        ready = os.path.join(td, "jobid")
        proc = subprocess.Popen(
            [sys.executable, "-c", CHILD, str(COMMON), base, "tok-secret", ready],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        deadline = time.time() + 20
        job_id = ""
        while time.time() < deadline:
            if os.path.exists(ready):
                job_id = open(ready).read().strip()
                break
            if proc.poll() is not None:
                print("FAIL 子进程提前退出：\n" + (proc.stdout.read() or ""))
                return 1
            time.sleep(0.05)
        if not job_id:
            proc.kill()
            print("FAIL 子进程未在 20s 内提交任务")
            return 1

        # 取消：模拟 Studio 终止执行 → 执行器发 SIGTERM
        proc.send_signal(signal.SIGTERM)
        try:
            out, _ = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            print("FAIL 子进程收到 SIGTERM 后未退出（优雅退出失效）")
            return 1
        srv.shutdown()
        output = out or ""

    failures = []
    if not received:
        failures.append("未收到 /interrupt：节点进程被终止时没有回收远程任务")
    else:
        if received[0]["body"].get("job_id") != job_id:
            failures.append(f"/interrupt job_id={received[0]['body'].get('job_id')} "
                            f"与提交的任务 {job_id} 不一致")
        if received[0]["auth"] != "Bearer tok-secret":
            failures.append(f"/interrupt 认证头错误：{received[0]['auth']!r}")
        if len(received) > 1:
            failures.append(f"/interrupt 重复发送 {len(received)} 次（应定向且只发一次）")
    if proc.returncode != 128 + signal.SIGTERM:
        failures.append(f"退出码 {proc.returncode}（期望 {128 + signal.SIGTERM}"
                        f"=信号处理器路径）")
    if len(jobs_submitted) != 1:
        failures.append(f"提交了 {len(jobs_submitted)} 个任务（期望 1）")

    if failures:
        print("FAIL")
        for f in failures:
            print("  - " + f)
        print("--- 子进程输出 ---\n" + output)
        return 1
    print(f"PASS 收到 SIGTERM → /interrupt(job_id={job_id[:8]}…) → 以 143 退出")
    return 0


if __name__ == "__main__":
    sys.exit(main())
