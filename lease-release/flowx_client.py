"""FlowX 节点共享 HTTP 客户端（纯标准库，零依赖）。
每个节点包通过 flowx.json 的 files 字段携带本文件。"""
import http.client
import json
import os
import time
import urllib.error
import urllib.parse
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
        # 隧道/代理的错误页不是服务端语义错误：cpolar 在隧道断开时会对任意路径回
        # 404/502 的 **HTML** 页（真机 exec 760：一分钟的隧道抖动被误判成
        # "job 不存在" ⇒ wait_job 立即致命，60 分钟的重绘任务白跑）。
        # 服务端（FastAPI）的错误体一律 JSON ⇒ 非 JSON 即视为**瞬态**，交上层重试。
        if body.lstrip()[:1] not in ("{", "["):
            raise RuntimeError(
                f"{method} {url} -> HTTP {e.code}（非 JSON 响应，疑似隧道/代理错误页，"
                f"瞬态可重试）: {body[:120].strip()}")
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


def delete_json(base, path, tok=None, timeout=60, retries=None, headers=None):
    """DELETE 幂等：瞬态错误自动重试（4xx 不重试）。lease-release 节点用。"""
    return _with_retry(lambda: json.loads(_req("DELETE", base, path, None, tok,
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


HEARTBEAT_SECONDS = float(os.environ.get("FLOWX_JOB_HEARTBEAT", "60"))


def flush_print(*args, **kwargs):
    """带 flush 的 print：节点日志走 docker exec 的**流式连接**，缓冲区攒着不刷
    ⇒ 流上长时间零字节 ⇒ 隧道判定空闲并截断（真机 exec 765：MaskRefine 在 docker 里
    跑 36 分钟、日志只落了 3 行，节点静默死亡且心跳一行都没出来）。"""
    kwargs.setdefault("flush", True)
    print(*args, **kwargs)


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
            if "-> HTTP 4" in str(e) and "非 JSON 响应" not in str(e):
                raise   # 服务端 4xx（JSON 错误体）= 请求本身的问题；隧道错误页除外
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


# ---------------------------------------------------------------------------
# WebSocket 事件等待（P1-16）：WS 优先、断线自动回退轮询
# ---------------------------------------------------------------------------
# 纯 stdlib 实现（RFC 6455 握手 + 帧解析），零依赖纪律不变。WS 只是"等结果
# 的方式"：任务状态机与终态判定仍以 REST 视图为准（终态事件到达后 GET 一次
# /jobs/{id} 取完整 result）。任何通道类问题（握手被拒/断开/读超时/帧异常）
# 抛 _WsFallback，由 wait_job 回退到原轮询路径——旧服务端（无 /ws）+ 新节点、
# 新服务端 + 旧节点两个方向都必须可用。FLOWX_WS=0 可整体关闭（一处 workaround
# 一个开关）。
_WS_DISABLED = os.environ.get("FLOWX_WS", "1").lower() in ("0", "false", "no")
_WS_READ_TIMEOUT = 45.0  # 服务端应用层 ping 间隔 30s + 15s 余量


class _WsFallback(Exception):
    """WS 通道不可用（握手失败/断开/读超时）：wait_job 捕获后回退轮询。"""


class _WsConn:
    """最小 WebSocket 连接：握手后的帧收（自动回 pong / 拼续帧 / 过滤 ping-pong）。
    只收不发（订阅由 URL 查询参数表达），唯一主动发送是 pong 与潜在 close。"""

    def __init__(self, sock, leftover=b""):
        self.sock = sock
        self.buf = bytearray(leftover)

    def _read(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise _WsFallback("连接被对端关闭")
            self.buf += chunk
        out = bytes(self.buf[:n])
        del self.buf[:n]
        return out

    def send_frame(self, opcode, payload=b""):
        """客户端发送必须掩码（RFC 6455 §5.3）；只用于 pong。"""
        import struct as _st
        mask = os.urandom(4)
        header = bytearray([0x80 | opcode])
        ln = len(payload)
        if ln < 126:
            header.append(0x80 | ln)
        elif ln < 65536:
            header += bytes([0x80 | 126]) + _st.pack(">H", ln)
        else:
            header += bytes([0x80 | 127]) + _st.pack(">Q", ln)
        self.sock.sendall(bytes(header) + mask
                          + bytes(b ^ mask[i % 4] for i, b in enumerate(payload)))

    def recv_event(self):
        """收一个完整文本消息并解析为 dict。协议 ping 自动回 pong；
        close 帧 / 连接断抛 _WsFallback。"""
        import struct as _st
        data = bytearray()
        while True:
            hdr = self._read(2)
            fin = hdr[0] & 0x80
            op = hdr[0] & 0x0F
            masked = hdr[1] & 0x80
            ln = hdr[1] & 0x7F
            if ln == 126:
                ln = _st.unpack(">H", self._read(2))[0]
            elif ln == 127:
                ln = _st.unpack(">Q", self._read(8))[0]
            if ln > 8 * 1024 * 1024:
                raise _WsFallback(f"帧过大: {ln}")
            mask = self._read(4) if masked else None
            payload = self._read(ln) if ln else b""
            if mask:
                payload = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
            if op == 0x9:    # ping → pong（uvicorn 默认 20s 协议级 ping，不回会被断）
                self.send_frame(0xA, payload)
                continue
            if op == 0xA:    # pong
                continue
            if op == 0x8:
                raise _WsFallback("收到 close 帧")
            if op in (0x0, 0x1, 0x2):
                data += payload
                if fin:
                    return json.loads(bytes(data))
            # 保留 opcode 忽略


def _ws_connect(base, tok, job_id, timeout=10):
    """WebSocket 握手（http→ws / https→wss 随 base 的 scheme）。
    成功返回 _WsConn；任何失败抛 _WsFallback。握手响应之后随包到达的
    帧字节（如 snapshot）收进 leftover 不丢。"""
    import base64 as _b64
    import socket as _sock
    import ssl as _ssl
    u = urllib.parse.urlparse(base)
    secure = u.scheme in ("https", "wss")
    host = u.hostname
    port = u.port or (443 if secure else 80)
    path = (u.path.rstrip("/") or "") + f"/ws?job_id={urllib.parse.quote(str(job_id), safe='')}"
    if tok:
        path += "&token=" + urllib.parse.quote(str(tok), safe="")
    try:
        s = _sock.create_connection((host, port), timeout=timeout)
        if secure:
            s = _ssl.create_default_context().wrap_socket(s, server_hostname=host)
        key = _b64.b64encode(os.urandom(16)).decode()
        s.sendall((f"GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\n"
                   "Upgrade: websocket\r\nConnection: Upgrade\r\n"
                   f"Sec-WebSocket-Key: {key}\r\n"
                   "Sec-WebSocket-Version: 13\r\n\r\n").encode())
        resp = b""
        while b"\r\n\r\n" not in resp:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
            if len(resp) > 65536:
                break
        head, _, leftover = resp.partition(b"\r\n\r\n")
        status = head.split(b"\r\n", 1)[0]
        if b" 101" not in status:
            raise _WsFallback(
                f"握手被拒: {status.decode(errors='replace')[:80]}")
        s.settimeout(_WS_READ_TIMEOUT)
        return _WsConn(s, leftover)
    except _WsFallback:
        raise
    except Exception as e:
        raise _WsFallback(str(e))


def _wait_job_ws(base, job_id, tok, timeout, on_poll, log):
    """WS 事件驱动等待。done 返回 result dict；任务 failed/cancelled/超时抛
    RuntimeError（与轮询路径同语义）；通道问题抛 _WsFallback（调用方回退轮询）。"""
    t0 = time.time()
    last = ""
    last_beat = time.time()
    conn = _ws_connect(base, tok, job_id)
    view = {"status": "pending", "progress": {"current": 0, "total": 0}}

    def _terminal():
        # 终态以 REST 视图为准（含完整 result/error；done 事件里的 result 同内容）
        try:
            v = get_job(base, job_id, tok)
        except RuntimeError as e:
            raise _WsFallback(f"终态确认失败: {e}")
        st = v.get("status")
        if st == "done":
            return v.get("result") or {}
        raise RuntimeError(f"job {job_id} {st}: {v.get('error')}")

    try:
        while True:
            try:
                ev = conn.recv_event()
            except _WsFallback:
                raise
            except Exception as e:  # socket.timeout / OSError / JSON 错
                raise _WsFallback(str(e))
            et = ev.get("type")
            if et == "ping":
                if log and time.time() - last_beat >= HEARTBEAT_SECONDS:
                    # 加载阶段无 progress 事件，靠服务端 30s ping 驱动心跳输出
                    # （docker 节点 stdout 保活，同轮询路径的 HEARTBEAT 语义）
                    p = view.get("progress") or {}
                    log(f"[job {job_id[:8]}] {view.get('status')} "
                        f"{p.get('current', 0)}/{p.get('total', 0)}"
                        f"（心跳 {time.time()-t0:.0f}s）")
                    last_beat = time.time()
                continue
            if et == "error":
                raise _WsFallback(str(ev.get("error") or "ws error"))
            if et == "snapshot":
                view = ev.get("job") or view
            elif et in ("submitted",):
                view["status"] = "pending"
            elif et == "running":
                view["status"] = "running"
            elif et == "progress":
                view["status"] = "running"
                view["progress"] = {"current": ev.get("current", 0),
                                    "total": ev.get("total", 0)}
            elif et == "stage":
                view["stage"] = {"index": ev.get("index"), "total": ev.get("total"),
                                 "node": ev.get("node"), "op": ev.get("op")}
            elif et in ("done", "failed", "cancelled"):
                return _terminal()
            else:
                continue
            if view.get("status") in ("done", "failed", "cancelled"):
                # snapshot 直接带终态（订阅时任务已结束）
                return _terminal()
            if on_poll is not None:
                try:
                    on_poll(view)
                except Exception:
                    pass
            p = view.get("progress") or {}
            line = f"[job {job_id[:8]}] {view.get('status')} " \
                   f"{p.get('current', 0)}/{p.get('total', 0)}"
            if log and line != last:
                log(line)
                last = line
                last_beat = time.time()
            if time.time() - t0 > timeout:
                try:
                    interrupt_job(base, job_id, tok)
                except RuntimeError:
                    pass
                raise RuntimeError(
                    f"job {job_id} timeout after {timeout}s (interrupted)")
    finally:
        _unregister_inflight(job_id)
        try:
            conn.sock.close()
        except Exception:
            pass


def wait_job(base, job_id, tok=None, timeout=7200, poll=5.0, log=flush_print,
             on_poll=None, max_poll_errors=12, error_window=300.0):
    """等待任务到终态：WS 事件优先（P1-16），WS 不可用自动回退轮询。
    done 返回 result dict；failed/cancelled/超时抛异常（超时先尽力 interrupt，
    避免孤儿任务继续占 GPU）。FLOWX_WS=0 强制走轮询。"""
    if not _WS_DISABLED:
        t0 = time.time()
        try:
            return _wait_job_ws(base, job_id, tok, timeout, on_poll, log)
        except _WsFallback as e:
            if log:
                log(f"[job {job_id[:8]}] ws 不可用（{e}），回退轮询")
            timeout = max(60.0, timeout - (time.time() - t0))
    return _wait_job_poll(base, job_id, tok, timeout, poll, log, on_poll,
                          max_poll_errors, error_window)


def _wait_job_poll(base, job_id, tok=None, timeout=7200, poll=5.0, log=flush_print,
             on_poll=None, max_poll_errors=12, error_window=300.0):
    # 心跳间隔（秒）：docker 节点靠 stdout 保活 docker exec 流（见下方注释）
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
    last_beat = time.time()
    errors = 0
    err_t0 = 0.0
    try:
        while True:
            try:
                v = get_job(base, job_id, tok)
                errors = 0
            except RuntimeError as e:
                msg = str(e)
                # 404 等**服务端** 4xx（JSON 错误体）说明 job 真不存在，重试无意义；
                # 但隧道/代理的 HTML 错误页必须走下面的瞬态重试（exec 760 教训）
                if "-> HTTP 4" in msg and "非 JSON 响应" not in msg:
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
                last_beat = time.time()
            elif log and time.time() - last_beat >= HEARTBEAT_SECONDS:
                # 心跳：docker 节点的 stdout 走 docker exec 的流式连接，隧道会把
                # **长时间无字节**的流截断（真机 exec 764：MaskRefine 在 docker 里跑
                # 47 分钟只发了两行，流被截断 ⇒ 节点失败而远程 job 还在跑）。
                # 周期性输出既保活又让日志能看到"还在跑"（本地节点多几行日志无副作用）。
                log(f"[job {job_id[:8]}] {st} {p.get('current', 0)}/{p.get('total', 0)}"
                    f"（心跳 {time.time()-t0:.0f}s）")
                last_beat = time.time()
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


def emit_input_media(base, oid, tok=None, name="input"):
    """把「输入原图」物化一份到 Studio 媒体缓存（kind=input，replace 式）。

    为什么需要：画布对比控件要在节点完成后并排显示「原图 / 结果」，而原图只存在于
    推理服务的对象存储（内存态）——GC、服务重启或对象 TTL 到期后
    {base}/images/{oid} 必 404，历史表现为「原图缩在左边 / 原图加载失败」。经 marker
    通道让 Studio 落盘一份后，原图与结果帧一样随媒体缓存（TTL 72h）持久可读。

    只对声明了 compareInput 的节点（widget-def）有意义；oid 缺失时静默跳过；
    同节点重复调用为覆盖式（只留最新一份）。"""
    if not oid or not isinstance(oid, str):
        return
    host = host_base(base)
    emit_media("input", url=f"{host}/images/{oid}", mode="replace", name=name,
               tok=tok, base=host)


def emit_preview(url, progress=None, tok=None, base=None, job_id=None):
    """经 stdout 标记通道向 Studio 上报预览帧地址（媒体本体不走 stdout/base64）：
    Studio 拦截 FLOWX_PREVIEW 行（不落日志），按 url 经 HTTP 中转拉帧给画布。
    url 指向推理服务的预览帧端点（应经 host_base() 构造，如
    {host_base}/preview/{job_id}）。
    base/job_id 供 Studio 记录服务基地址与推理 job（节点级中断、
    op-replay 重放预览用）；旧版 Studio 忽略多余字段。

    路线 A（媒体通道统一化）：同时发 FLOWX_MEDIA（新格式）。kind=frame +
    mode=replace 语义与旧 FLOWX_PREVIEW 等价；旧 Studio 只识别 FLOWX_PREVIEW，
    FLOWX_MEDIA 行会落入普通日志（无害）。"""
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
    emit_media("frame", url=url, mode="replace", progress=payload.get("progress"),
               tok=tok, base=base, job_id=job_id)


def emit_media(kind, url=None, path=None, mode=None, name=None, mime=None,
               progress=None, tok=None, base=None, job_id=None):
    """上报节点媒体产物（路线 A 新格式）。source：url（http，Studio 物化拉取落盘）
    或 path（本地文件，登记不复制，须落在媒体白名单内）。kind 是不透明分类标签
    （frame/result/... 由节点生态自描述）；mode=replace 覆盖式最新 / append 追加序列。"""
    m = {"kind": kind}
    if mode:
        m["mode"] = mode
    if name:
        m["name"] = name
    if mime:
        m["mime"] = mime
    src = {}
    if url:
        src["type"] = "http"
        src["url"] = url
        if tok:
            src["token"] = tok
    elif path:
        src["type"] = "local"
        src["path"] = path
    if src:
        m["source"] = src
    if progress is not None:
        m["progress"] = round(float(progress), 4)
    if base:
        m["base"] = base
    if job_id:
        m["job_id"] = job_id
    print("FLOWX_MEDIA " + json.dumps(m, separators=(",", ":")), flush=True)
