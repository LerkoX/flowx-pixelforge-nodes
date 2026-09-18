"""FlowX 节点共享 HTTP 客户端（纯标准库，零依赖）。
每个节点包通过 flowx.json 的 files 字段携带本文件。"""
import json
import os
import time
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
    except (TimeoutError, OSError) as e:
        # 读取响应体阶段的 socket.timeout（TimeoutError）/连接重置不是 URLError，
        # 隧道卡顿时连接已建立但传输停滞即为此类；同样视为瞬态错误交给上层重试。
        # （URLError 本身是 OSError 子类，已在上方捕获，此分支只接非 URLError 的）
        raise RuntimeError(f"{method} {url} failed: {type(e).__name__}: {e}")


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
    """调能力运行时 POST /op（同步），返回展平的 {端口: id或值}。"""
    resp = post_json(base, "/op", {"name": name, "inputs": inputs}, tok, timeout)
    return {k: m.get("id", m.get("value"))
            for k, m in resp.get("outputs", {}).items()}


def submit_job(base, payload, tok=None, timeout=60):
    """提交异步任务：payload = {"nodes": ...} 或 {"name", "inputs"}；返回 job_id。"""
    return post_json(base, "/jobs", payload, tok, timeout)["job_id"]


def get_job(base, job_id, tok=None, timeout=30):
    """查询任务状态/进度；done 时带 result。"""
    return get_json(base, f"/jobs/{job_id}", tok, timeout)


def interrupt_job(base, job_id=None, tok=None, timeout=30):
    """取消任务：带 job_id 取消指定任务；None 取消全部 running+pending。"""
    payload = {"job_id": job_id} if job_id else {}
    return post_json(base, "/interrupt", payload, tok, timeout)


def wait_job(base, job_id, tok=None, timeout=7200, poll=5.0, log=print,
             on_poll=None, max_poll_errors=12):
    """轮询任务直到终态。done 返回 result dict；failed/cancelled/超时抛异常
    （超时先尽力 interrupt，避免孤儿任务继续占 GPU）。
    on_poll(view)：每次轮询拿到任务视图后回调（如上报预览帧地址），
    回调异常静默忽略，不影响任务等待。
    轮询容错：瞬态网络错误（隧道抖动/超时/连接重置）重试，连续
    max_poll_errors 次（默认 ≈ 1 分钟窗口）才放弃；HTTP 4xx（如 job
    不存在）立即失败，不重试。"""
    t0 = time.time()
    last = ""
    errors = 0
    while True:
        try:
            v = get_job(base, job_id, tok)
            errors = 0
        except RuntimeError as e:
            msg = str(e)
            if "-> HTTP 4" in msg:  # 404 等：job 真不存在，重试无意义
                raise
            errors += 1
            if log:
                log(f"[job {job_id[:8]}] poll error ({errors}/{max_poll_errors}): {msg}")
            if errors >= max_poll_errors:
                raise RuntimeError(
                    f"job {job_id} poll failed {errors} times in a row, last: {msg}")
            time.sleep(poll)
            continue
        if on_poll is not None:
            try:
                on_poll(v)
            except Exception:
                pass
        st = v.get("status")
        p = v.get("progress") or {}
        line = f"[job {job_id[:8]}] {st} {p.get('current', 0)}/{p.get('total', 0)}"
        if log and line != last:
            log(line)
            last = line
        if st == "done":
            return v.get("result") or {}
        if st in ("failed", "cancelled"):
            raise RuntimeError(f"job {job_id} {st}: {v.get('error')}")
        if time.time() - t0 > timeout:
            try:
                interrupt_job(base, job_id, tok)
            except RuntimeError:
                pass
            raise RuntimeError(f"job {job_id} timeout after {timeout}s (interrupted)")
        time.sleep(poll)


def ref(oid):
    """构造对象端口引用。"""
    return {"$id": oid}


def emit(**fields):
    """以 flowx-yaml 代码块输出结果字段，供 FlowX 提取为 Metadata。"""
    print("```flowx-yaml")
    for k, v in fields.items():
        print(f'{k}: "{v}"')
    print("```")


def emit_preview(url, progress=None, tok=None):
    """经 stdout 标记通道向 Studio 上报预览帧地址（媒体本体不走 stdout/base64）：
    Studio 拦截 FLOWX_PREVIEW 行（不落日志），按 url 经 HTTP 中转拉帧给画布。
    url 指向推理服务的预览帧端点（如 {service_url}/preview/{job_id}）。"""
    payload = {"url": url}
    if progress is not None:
        payload["progress"] = round(float(progress), 4)
    if tok:
        payload["token"] = tok
    print("FLOWX_PREVIEW " + json.dumps(payload, separators=(",", ":")),
          flush=True)
