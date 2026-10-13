"""FlowX 节点共享 HTTP 客户端（纯标准库，零依赖）。
每个节点包通过 flowx.json 的 files 字段携带本文件。"""
import http.client
import json
import os
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# 优雅退出：节点进程被终止时，回收自己提交的远程（推理服务）后台任务
# ---------------------------------------------------------------------------
# 触发路径：Studio「终止执行」/节点超时 → flowx 本地执行器对整棵进程树发 SIGTERM，
# 宽限期（默认 10s，执行器 config.interruptGrace）内不退出才 SIGKILL。节点脚本必须
# 在这个窗口内取消自己在服务端 running 的 job，否则远程任务变孤儿继续占 GPU。
# 同时接管 SIGHUP：PTY 路径（script/bash 包装，Studio 默认执行器 pty=true）下包装
# 进程先退出会关闭终端，内核补发 SIGHUP——不接管就会被默认动作杀死，清理跑不了一半。
# SIGKILL（宽限期用尽/进程被直接强杀）无法捕获，此时依赖 Studio 侧按节点上报的
# job_id 转发 /interrupt 兜底（见 flowx-studio CancelExecution）。
_INFLIGHT = {}  # job_id -> (base, tok)：submit_job 登记，wait_job 终态注销

# F1（dev-plan §47）：基名 → 服务端实际算子名（`<基名>@<族内容hash12>`）。
# ensure_plugin 解析成功后登记；call_op/submit_job 自动翻译——这样 60 个自包含节点
# 里写死的字面算子名（call_op(url, "sd.sample", …)）无需逐个改。
_OP_ALIAS = {}


def op_alias(name):
    """基名 → 实际算子名；未登记的（核心/历史算子）原样返回。"""
    return _OP_ALIAS.get(name, name)
_SIGNAL_HANDLERS_READY = False


def _register_inflight(base, job_id, tok):
    """登记进行中的远程任务，并确保终止信号处理器已安装。
    信号只能在主线程注册；注册失败（非主线程/受限环境）时退化为依赖 Studio 侧
    兜底转发 /interrupt。"""
    global _SIGNAL_HANDLERS_READY
    _INFLIGHT[job_id] = (base, tok)
    if _SIGNAL_HANDLERS_READY:
        return
    import signal
    import sys

    def _on_signal(signum, _frame):
        try:
            signame = signal.Signals(signum).name
        except ValueError:
            signame = str(signum)
        print(f"[flowx_client] 收到 {signame}：取消进行中的远程任务 "
              f"({len(_INFLIGHT)} 个) 后退出", flush=True)
        for job_id, (base, tok) in list(_INFLIGHT.items()):
            _INFLIGHT.pop(job_id, None)
            try:
                # 短超时必须落在执行器宽限期内，否则会被 SIGKILL 打断
                interrupt_job(base, job_id, tok, timeout=5)
                print(f"[flowx_client] 已取消远程任务 {job_id[:8]}", flush=True)
            except Exception as e:  # 尽力而为：清理失败不阻塞退出
                print(f"[flowx_client] 取消远程任务 {job_id[:8]} 失败: {e}",
                      flush=True)
        sys.exit(128 + signum)

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _on_signal)
        except (ValueError, OSError, AttributeError):
            # 非主线程（无法注册信号）或平台无此信号：跳过，由 Studio 侧兜底
            pass
    _SIGNAL_HANDLERS_READY = True


def _unregister_inflight(job_id):
    _INFLIGHT.pop(job_id, None)


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


def lease_id():
    """执行租约标识（dev-plan §43.2）：Studio 注入的 FLOWX_EXECUTION_ID。
    非 Studio 环境（命令行直跑/mock）为空 → 不加头，服务端行为不变。"""
    return os.environ.get("FLOWX_EXECUTION_ID") or ""


def _req(method, base, path, payload=None, tok=None, timeout=1800,
         content_type="application/json", headers=None):
    """payload 为 bytes（已编码）；GET/无 body 时传 None。

    headers：调用方显式附加的请求头（**覆盖**同名自动头，如
    `{"X-FlowX-Lease": "724"}` 以指定租约视角清理别人留下的常驻条目，
    见 vram-ops 节点与 dev-log §58.9）。"""
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, data=payload, method=method)
    if payload is not None:
        req.add_header("Content-Type", content_type)
    if tok:
        req.add_header("Authorization", "Bearer " + tok)
    lid = lease_id()
    if lid:
        # 执行租约（§43.2）：加载/淘汰按租约隔离，并发执行排队而非互踩
        req.add_header("X-FlowX-Lease", lid)
    for hk, hv in (headers or {}).items():
        req.add_header(hk, hv)   # 显式头最后加：同名时覆盖自动注入值
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


# 瞬态重试次数（dev-plan §43.5 遗留：隧道瞬断率 ~50% 时 GET /ops 等单发
# 调用瞬败导致节点假失败）。仅作用于幂等调用；FLOWX_HTTP_RETRIES=1 关闭。
HTTP_RETRIES = int(os.environ.get("FLOWX_HTTP_RETRIES", "3"))


def _retriable(err):
    """HTTP 4xx 是请求本身的问题，重试无意义；超时/连接重置/EOF/5xx 视为瞬态。"""
    return "-> HTTP 4" not in str(err)



def _with_retry(fn, retries=None):
    """对瞬态错误重试（线性退避，对齐 call_op）。仅用于幂等调用（GET、
    内容寻址的幂等上传）；非幂等调用（如 POST /jobs 提交任务，重试会
    产生重复任务）不得走这里。

    B4/S3 修正：隧道下只读回一半 body 会抛 `http.client.IncompleteRead`
    （`HTTPException` 子类），而旧实现只捕 `RuntimeError` ⇒ 直接失败
    （真机在 `GET /ops` 上踩过两次）。改为捕获 `(RuntimeError, OSError,
    http.client.HTTPException)`；`URLError`/`ConnectionError` 都是 OSError 子类。
    """
    n = max(1, int(retries if retries is not None else HTTP_RETRIES))
    last = None
    for attempt in range(1, n + 1):
        try:
            return fn()
        except (RuntimeError, OSError, http.client.HTTPException) as e:
            last = e
            if not _retriable(e) or attempt >= n:
                raise
            time.sleep(min(5 * attempt, 15))
    raise last



def post_json(base, path, payload, tok=None, timeout=1800, headers=None):
    data = json.dumps(payload).encode()
    return json.loads(_req("POST", base, path, data, tok, timeout,
                           headers=headers))


def post_bytes(base, path, data, tok=None, timeout=300,
               content_type="application/octet-stream", headers=None):
    """上传原始字节（如图片文件），返回解析后的 JSON 响应。"""
    return json.loads(_req("POST", base, path, data, tok, timeout, content_type,
                           headers=headers))


def get_json(base, path, tok=None, timeout=60, retries=None, headers=None):
    """GET 幂等：瞬态错误自动重试（隧道抖动兜底，4xx 不重试）。"""
    return _with_retry(lambda: json.loads(_req("GET", base, path, None, tok,
                                               timeout, headers=headers)),
                       retries)


def get_bytes(base, path, tok=None, timeout=300, retries=None):
    """GET 幂等：瞬态错误自动重试（隧道抖动兜底，4xx 不重试）。"""
    return _with_retry(lambda: _req("GET", base, path, None, tok, timeout),
                       retries)


def plugin_identity(plugin_path):
    """读节点本地 server_op.py 的族版本化常量（F1，dev-plan §47）。

    返回 (family, version)：族文件带 `OP_FAMILY`/`OP_VERSION` 时给出落盘名前缀与版本；
    未改造的旧族文件两者都无（返回 (None, None)，沿用旧命名规则）。
    只解析顶层字面量赋值，不 import（无副作用、不需族文件依赖）。
    """
    import ast
    try:
        with open(plugin_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (OSError, SyntaxError):
        return None, None
    vals = {}
    for node in tree.body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("OP_FAMILY", "OP_VERSION")
                and isinstance(node.value, ast.Constant)):
            vals[node.targets[0].id] = node.value.value
    fam, ver = vals.get("OP_FAMILY"), vals.get("OP_VERSION")
    if fam is None and ver is None:
        return None, None          # 旧族文件：兼容
    if not (isinstance(fam, str) and isinstance(ver, str) and fam and ver):
        raise RuntimeError(
            f"{plugin_path}: OP_FAMILY/OP_VERSION 必须成对且为非空字符串"
            f"（F1 族版本化合同，见 dev-plan §47）；请跑 nodes/_tools/sync-common.py")
    return fam, ver


def _resolve_op_name(ops, op_name, digest):
    """在 /ops 里找「同基名 + 本文件内容 hash」的实际算子名。

    F1 之后族注册名是 `<基名>@<族内容hash12>`，故实际名与基名不同；
    旧族文件（未版本化）注册名就等于基名。同名归并只发生在内容相同时（语义等价）。
    """
    for o in ops:
        n = o.get("name") or ""
        if (n == op_name or n.startswith(op_name + "@")) \
                and o.get("plugin_hash") == digest:
            return n
    return None


def ensure_plugin(base, op_name, plugin_path=None, tok=None, timeout=120):
    """节点算子自注册：确保服务端已注册 op_name 且与本地插件文件版本一致，
    缺失或 hash 不符则经 POST /admin/plugins 上传热加载（服务端需配置
    INFERENCE_TOKEN 开启 /admin/*，且节点参数 service_token 填入同一令牌）。
    plugin_path 缺省为节点包内的 server_op.py（与 main.py 同目录）。
    算子已存在且为核心内置（无 plugin_hash）时视为可用。

    **返回实际算子名**：F1 之后族文件把注册名写成 `op_name@<族内容hash12>`，
    调用方必须用返回名提交 job（否则算子不存在）——这也正是「每执行钉在自己那版
    实现上」的机制（旧版顶不掉新版，见 dev-plan §47）。
    """
    import hashlib
    if plugin_path is None:
        plugin_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "server_op.py")
    with open(plugin_path, "rb") as f:
        content = f.read()
    digest = hashlib.sha256(content).hexdigest()

    # 落盘名：族文件内容寻址（<OP_FAMILY>_<OP_VERSION>.py）——同内容=同文件=幂等，
    # 不同内容=不同文件=新旧版本并存；旧族文件沿用 <op_name>.py 命名。
    family, version = plugin_identity(plugin_path)
    filename = (f"{family}_{version}.py" if family
                else op_name.replace(".", "_") + ".py")

    def _done(actual):
        """登记别名并返回实际算子名（F1：调用方/后续 call_op 都按它提交）。"""
        if actual != op_name:
            _OP_ALIAS[op_name] = actual
        return actual

    ops = get_json(base, "/ops", tok, timeout=30).get("ops", [])
    ready = _resolve_op_name(ops, op_name, digest)
    if ready:
        return _done(ready)       # 本内容已注册（可能是版本化名）：就绪
    for o in ops:
        if o.get("name") == op_name and o.get("plugin_hash") is None:
            return _done(op_name)  # 核心内置：无插件身份

    payload = {"filename": filename,
               "content": content.decode("utf-8"), "sha256": digest}

    def _op_ready():
        """■入后复查：本内容是否已被任何同基名算子注册（并发上传竞态收敛判据）。

        扇出场景（如 10 个预处理器并行）各自 ensure_plugin：第一个上传成功后，
        其余节点会在上方的预检里命中并直接返回；仍有极小窗口撞上并发写
        （409/400）——只要求到的实际名字与本地实现一致，语义上就已就绪。
        """
        return _resolve_op_name(
            get_json(base, "/ops", tok, timeout=30).get("ops", []),
            op_name, digest)

    # /admin/plugins 按内容 sha256 上传：同内容同文件幂等，可安全重试
    last_err = None
    for attempt in (1, 2):
        try:
            resp = _with_retry(lambda: post_json(base, "/admin/plugins", payload,
                                                 tok, timeout))
        except RuntimeError as e:
            msg = str(e)
            if "-> HTTP 404" in msg:
                raise RuntimeError(
                    f"算子 '{op_name}' 未注册且服务端 /admin 未开放"
                    f"（INFERENCE_TOKEN 未配置）。请在服务端配置 token 并在节点"
                    f" service_token 参数填入同一值后重试") from e
            if "-> HTTP 409" in msg or "-> HTTP 400" in msg:
                if attempt == 1:
                    time.sleep(1.5)   # 等同族并发上传落地
                    ready = _op_ready()
                    if ready:
                        print(f"[ensure_plugin] {op_name} 已由同内容插件注册"
                              f"（并发上传竞态，跳过重传）", flush=True)
                        return _done(ready)
                    last_err = e
                    continue
                ready = _op_ready()
                if ready:
                    print(f"[ensure_plugin] {op_name} 已由同内容插件注册"
                          f"（并发上传竞态，跳过重传）", flush=True)
                    return _done(ready)
                raise RuntimeError(
                    f"{op_name} 插件上传被拒（{msg[:160]}）：服务端可能已有**不同"
                    f"内容**的插件文件持有同基名算子（plugins.d 下同基名文件冲突）。"
                    f"处理：删除该插件文件（DELETE /admin/plugins/<file>）或重启"
                    f"服务端后重试") from e
            raise
        registered = resp.get("ops", [])
        actual = _resolve_op_name(
            [{"name": n, "plugin_hash": digest} for n in registered],
            op_name, digest)
        if not actual:
            raise RuntimeError(
                f"插件已上传但算子 '{op_name}' 未注册"
                f"（插件实际注册了 {registered}）")
        if actual != op_name:
            print(f"[ensure_plugin] {op_name} -> {actual}"
                  f"（{resp.get('plugin')}）", flush=True)
        else:
            print(f"[ensure_plugin] {op_name} 自注册成功"
                  f"（{resp.get('plugin')}）", flush=True)
        return _done(actual)
    raise last_err


def call_op(base, name, inputs, tok=None, timeout=1800, max_retries=3):
    """调能力运行时 POST /op（同步），返回展平的 {端口: id或值}。
    瞬态网络错误重试 max_retries 次：/op 算子多为幂等计算（结果进对象仓库），
    超时断开时服务端已完成执行，重试仅重算一次，无副作用。"""
    name = op_alias(name)  # F1 基名→实际名（未登记的原样通行）
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
    """提交异步任务：payload = {"nodes": ...} 或 {"name", "inputs"}；返回 job_id。
    任务登记为进行中：进程收到终止信号（Studio 取消/超时）时会被定向 interrupt，
    避免节点被终止后远程任务变孤儿。"""
    if isinstance(payload, dict) and payload.get("name"):
        payload = dict(payload, name=op_alias(payload["name"]))  # F1 基名→实际名
    job_id = post_json(base, "/jobs", payload, tok, timeout)["job_id"]
    _register_inflight(base, job_id, tok)
    return job_id


def get_job(base, job_id, tok=None, timeout=30):
    """查询任务状态/进度；done 时带 result。"""
    return get_json(base, f"/jobs/{job_id}", tok, timeout)


def interrupt_job(base, job_id=None, tok=None, timeout=30):
    """取消任务：带 job_id 取消指定任务；None 取消全部 running+pending。"""
    payload = {"job_id": job_id} if job_id else {}
    return post_json(base, "/interrupt", payload, tok, timeout)


def wait_job(base, job_id, tok=None, timeout=7200, poll=5.0, log=print,
             on_poll=None, max_poll_errors=12, error_window=300.0):
    """轮询任务直到终态。done 返回 result dict；failed/cancelled/超时抛异常
    （超时先尽力 interrupt，避免孤儿任务继续占 GPU）。
    on_poll(view)：每次轮询拿到任务视图后回调（如上报预览帧地址），
    回调异常静默忽略，不影响任务等待。
    轮询容错：瞬态网络错误（隧道抖动/超时/连接重置/SSL EOF）重试，
    连续失败 ≥max_poll_errors 次【且】持续 ≥error_window 秒（默认 5 分钟）
    才放弃——双条件避免长任务（FLUX 采样 30min+）期间隧道长时间断流
    被误判为任务失败；HTTP 4xx（如 job 不存在）立即失败，不重试。"""
    t0 = time.time()
    last = ""
    errors = 0
    err_t0 = 0.0
    try:
        while True:
            try:
                v = get_job(base, job_id, tok)
                errors = 0
            except RuntimeError as e:
                msg = str(e)
                if "-> HTTP 4" in msg:  # 404 等：job 真不存在，重试无意义
                    raise
                if errors == 0:
                    err_t0 = time.time()
                errors += 1
                elapsed = time.time() - err_t0
                if log:
                    log(f"[job {job_id[:8]}] poll error ({errors}, "
                        f"{elapsed:.0f}s/{error_window:.0f}s): {msg}")
                if errors >= max_poll_errors and elapsed >= error_window:
                    raise RuntimeError(
                        f"job {job_id} poll failed {errors} times over "
                        f"{elapsed:.0f}s, last: {msg}")
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
    finally:
        # 终态（或异常）：从进行中表注销，避免终止信号重复取消已结束的任务
        _unregister_inflight(job_id)


def ref(oid):
    """构造对象端口引用。"""
    return {"$id": oid}


def yaml_scalar(v):
    """值 → 合法 YAML 双引号标量（供 flowx-yaml 块）。

    JSON 转义是 YAML 双引号标量的子集，所以 `json.dumps(str(v))` 对含 `"`、`:`、
    换行的值都安全 —— 早期两处裸拼 `f'{k}: "{v}"'`（本函数与 executor_base）
    会在 JSON 字符串输出上把 YAML 解析打断（B3 真机复现 __extract_missing）。
    """
    return json.dumps(str(v), ensure_ascii=False)


def emit(**fields):
    """以 flowx-yaml 代码块输出结果字段，供 FlowX 提取为 Metadata。"""
    print("```flowx-yaml")
    for k, v in fields.items():
        print(f"{k}: {yaml_scalar(v)}")
    print("```")


def host_base(url):
    """Studio 宿主机侧可达的服务基地址（画布预览帧/重放/中断中转都由 Studio
    发起 HTTP 请求，用的是 Studio 自己的网络视角）。

    容器内可达 ≠ Studio 可达：docker 节点跑在远端宿主上，`service_url` 只能填
    docker bridge 网关（如 http://172.17.0.1:8100），而 Studio 在手机/外网侧
    只能走公网隧道——此时必须由 workflow 给节点绑 `service_url_host`
    （如 {{ Param.service_url }}），否则画布取不到预览帧。

    未配置时退回 url（local 节点或 Studio 与推理服务同网段的部署下二者相同）。"""
    host = (os.environ.get("SERVICE_URL_HOST")
            or os.environ.get("FLOWX_PARAM_SERVICE_URL_HOST") or "")
    return host.rstrip("/") or url


def emit_result_preview(base, oid, tok=None):
    """结果对象完成后上报一张预览帧（Studio 经 /preview-frame 中转拉取
    {base}/images/{oid}）。媒体走 HTTP 二进制通道，**不得**把图片 base64 塞进
    outputs/metadata（画布 widget 读 props.preview 渲染）。"""
    host = host_base(base)
    emit_preview(f"{host}/images/{oid}", 1.0, tok, base=host)


def emit_preview(url, progress=None, tok=None, base=None, job_id=None):
    """经 stdout 标记通道向 Studio 上报预览帧地址（媒体本体不走 stdout/base64）：
    Studio 拦截 FLOWX_PREVIEW 行（不落日志），按 url 经 HTTP 中转拉帧给画布。
    url 指向推理服务的预览帧端点（应经 host_base() 构造，如
    {host_base}/preview/{job_id}）。
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
