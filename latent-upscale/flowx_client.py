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


def ensure_plugin(base, op_name, plugin_path=None, tok=None, timeout=120):
    """节点算子自注册：确保服务端已注册 op_name 且与本地插件文件版本一致，
    缺失或 hash 不符则经 POST /admin/plugins 上传热加载（服务端需配置
    INFERENCE_TOKEN 开启 /admin/*，且节点参数 service_token 填入同一令牌）。
    plugin_path 缺省为节点包内的 server_op.py（与 main.py 同目录）。
    算子已存在且为核心内置（无 plugin_hash）时视为可用，直接返回。"""
    import hashlib
    if plugin_path is None:
        plugin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "server_op.py")
    with open(plugin_path, "rb") as f:
        content = f.read()
    digest = hashlib.sha256(content).hexdigest()

    ops = get_json(base, "/ops", tok, timeout=30).get("ops", [])
    for o in ops:
        if o.get("name") != op_name:
            continue
        if o.get("plugin_hash") is None or o.get("plugin_hash") == digest:
            return  # 核心内置或插件版本一致：就绪
        break  # 插件版本不符：走上传覆盖

    payload = {"filename": op_name.replace(".", "_") + ".py",
               "content": content.decode("utf-8"), "sha256": digest}
    try:
        resp = post_json(base, "/admin/plugins", payload, tok, timeout)
    except RuntimeError as e:
        if "-> HTTP 404" in str(e):
            raise RuntimeError(
                f"算子 '{op_name}' 未注册且服务端 /admin 未开放"
                f"（INFERENCE_TOKEN 未配置）。请在服务端配置 token 并在节点"
                f" service_token 参数填入同一值后重试") from e
        raise
    registered = resp.get("ops", [])
    if op_name not in registered:
        raise RuntimeError(
            f"插件已上传但算子 '{op_name}' 未注册（插件实际注册了 {registered}）")
    print(f"[ensure_plugin] {op_name} 自注册成功（{resp.get('plugin')}）",
          flush=True)


def call_op(base, name, inputs, tok=None, timeout=1800, max_retries=3):
    """调能力运行时 POST /op（同步），返回展平的 {端口: id或值}。
    瞬态网络错误重试 max_retries 次：/op 算子多为幂等计算（结果进对象仓库），
    超时断开时服务端已完成执行，重试仅重算一次，无副作用。"""
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = post_json(base, "/op", {"name": name, "inputs": inputs},
                             tok, timeout)
            return {k: m.get("id", m.get("value"))
                    for k, m in resp.get("outputs", {}).items()}
        except RuntimeError as e:
            last_err = e
            if "-> HTTP 4" in str(e):  # 4xx 是请求本身的问题，重试无意义
                raise
            time.sleep(min(5 * attempt, 15))
    raise last_err


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


def emit_preview(url, progress=None, tok=None, base=None, job_id=None):
    """经 stdout 标记通道向 Studio 上报预览帧地址（媒体本体不走 stdout/base64）：
    Studio 拦截 FLOWX_PREVIEW 行（不落日志），按 url 经 HTTP 中转拉帧给画布。
    url 指向推理服务的预览帧端点（如 {service_url}/preview/{job_id}）。
    base/job_id 供 Studio 记录服务基地址与推理 job（节点级中断、
    op-replay 重放预览用）；旧版 Studio 忽略多余字段。"""
    payload = {"url": url}
    if progress is not None:
        payload["progress"] = round(float(progress), 4)
    if tok:
        payload["token"] = tok
    if base:
        payload["base"] = base
    if job_id:
        payload["job_id"] = job_id
    print("FLOWX_PREVIEW " + json.dumps(payload, separators=(",", ":")),
          flush=True)
