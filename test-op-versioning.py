#!/usr/bin/env python3
"""F1 算子身份版本化 单测（dev-plan §47）：不依赖推理服务、不需要 torch。

做法：把 `_common/flowx_client.py` 的 HTTP 层（`_req`）换成**假服务端**，并让假服务端
**真的 exec 族文件**（用可记录注册名的假 registry）——因此测的是真族文件里的
`_VersionedRegistry` 包装 + 真 `ensure_plugin` 解析逻辑，而不是复刻实现。

覆盖：
  1. plugin_identity 解析（版本化族 / 旧族 / 半残常量报错）
  2. 版本化上传：落盘名 <OP_FAMILY>_<ver>.py，返回 <基名>@<ver>
  3. **F1 核心**：改过实现的 v2 上传后，v1 与 v2 在服务端【并存】；
     再跑一次 v1（老节点后跑）**不会顶掉 v2**（这正是 F1 要根治的旧版覆盖）
  4. 同内容重传幂等（不上传第二次）
  5. 旧族文件（无 OP_FAMILY/OP_VERSION）沿用旧命名 + 基名
  6. call_op / submit_job 自动把基名翻译成实际名
  7. 跨文件不同内容抢同基名 → 明确报错（而不是静默顶掉）
用法：python3 nodes/test-op-versioning.py
"""
import importlib.util
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "_tools"))

FAIL = 0


def check(label, cond, extra=""):
    global FAIL
    print(("PASS " if cond else "FAIL ") + label + (f"  {extra}" if extra else ""),
          flush=True)
    if not cond:
        FAIL += 1


# ---------- 加载被测客户端 ----------
spec = importlib.util.spec_from_file_location("flowx_client",
                                              ROOT / "_common" / "flowx_client.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)
_sc = importlib.util.spec_from_file_location(
    "sync_common", ROOT / "_tools" / "sync-common.py")
sync_common = importlib.util.module_from_spec(_sc)
_sc.loader.exec_module(sync_common)   # 复用 sync-common 的打戳口径（单一事实源）

# ---------- 假服务端（含插件注册表语义）----------
STAMP_RE = re.compile(r'^OP_VERSION\s*=\s*"([0-9a-f]+)"', re.M)


class FakeService:
    """最小 /ops + /admin/plugins + /jobs：与真实服务端同语义（按文件名归属、409 跨文件重名）。"""

    def __init__(self):
        self.files = {}        # filename -> content(str)
        self.ops = {}          # op_name -> {plugin_hash, plugin_file}
        self.uploads = 0

    # --- 插件加载：真 exec 族文件，用记录型假 registry ---
    def _load(self, filename, content):
        recorded = []

        class FakeRegistry:
            def register(reg_self, name, inputs=None, outputs=None, description=""):
                def deco(fn=None):
                    recorded.append(name)
                    return fn
                return deco

        ns = {"__name__": f"fake_plugin_{filename}"}
        exec(compile(content, filename, "exec"), ns)
        ns["register"](FakeRegistry())   # 族文件的 register(registry)（内部做版本化包装）
        return sorted(set(recorded))

    def handle(self, method, path, payload):
        import hashlib
        import json as _json
        if method == "GET" and path == "/ops":
            return {"ops": [dict(name=n, **v) for n, v in self.ops.items()]}
        if method == "POST" and path == "/admin/plugins":
            fn = payload["filename"]
            raw = payload["content"].encode()
            if hashlib.sha256(raw).hexdigest() != payload["sha256"]:
                raise RuntimeError("POST /admin/plugins -> HTTP 400: sha256 mismatch")
            claimed = self._load(fn, payload["content"])
            conflicts = [n for n in claimed
                         if n in self.ops and self.ops[n]["plugin_file"] != fn]
            if conflicts:
                raise RuntimeError(
                    'POST /admin/plugins -> HTTP 409: {"detail":{"error":"op '
                    f'conflicts (rolled back)","conflicts":{conflicts}}}')
            self.files[fn] = payload["content"]
            self.uploads += 1
            for n in claimed:
                self.ops[n] = {"plugin_hash": payload["sha256"], "plugin_file": fn}
            return {"plugin": fn, "ops": claimed, "sha256": payload["sha256"]}
        raise RuntimeError(f"{method} {path} -> HTTP 404: not found")

    def hook(self, method, base, path, payload=None, tok=None, timeout=None,
             content_type=None):
        import json as _json
        out = self.handle(method, path, _json.loads(payload) if payload else None)
        return _json.dumps(out).encode()


svc = FakeService()
fc._req = svc.hook
fc._with_retry = lambda fn, retries=None: fn()
fc.get_json = lambda base, path, tok=None, timeout=60, retries=None: \
    __import__("json").loads(svc.hook("GET", base, path))
fc.post_json = lambda base, path, payload, tok=None, timeout=1800: \
    __import__("json").loads(svc.hook("POST", base, path, __import__("json").dumps(payload)))
fc.time.sleep = lambda *_: None      # 竞态退避不等真时间

BASE = "http://fake"


def use_service():
    """把客户端 HTTP 层切回假服务端（post_json/get_json 替身会互相覆盖）。"""
    import json as _json
    fc._req = svc.hook
    fc.get_json = lambda base, path, tok=None, timeout=60, retries=None: \
        _json.loads(svc.hook("GET", base, path))
    fc.post_json = lambda base, path, payload, tok=None, timeout=1800: \
        _json.loads(svc.hook("POST", base, path, _json.dumps(payload)))
FAM = ROOT / "_common" / "server_ops" / "preprocess_g7.py"
FAM_TEXT = FAM.read_text()
V1 = STAMP_RE.search(FAM_TEXT).group(1)

print("=== 1) plugin_identity 解析 ===")
import tempfile
_tmpdir = tempfile.TemporaryDirectory(prefix="flowx-f1-test-")
tmp = pathlib.Path(_tmpdir.name)   # 系统临时目录，测试不在仓库里留垃圾
p1 = tmp / "server_op_v1.py"
p1.write_text(FAM_TEXT)
fam, ver = fc.plugin_identity(str(p1))
check("版本化族：解析出 OP_FAMILY/OP_VERSION", (fam, ver) == ("preprocess_g7", V1),
      f"{fam}@{ver}")
p_legacy = tmp / "server_op_legacy.py"
# 真实"旧族文件" = F1 之前的形态：既无常量，也无包装（连注册名都是基名）
_f1_start = FAM_TEXT.index("# --- F1 族版本化")
_f1_end = FAM_TEXT.index("# ---------------------------------------------------------------------------",
                         _f1_start) + len("# ---------------------------------------------------------------------------\n")
LEGACY_TEXT = (FAM_TEXT[:_f1_start] + FAM_TEXT[_f1_end:]).replace(
    "    registry = _VersionedRegistry(registry, OP_VERSION)\n", "")
p_legacy.write_text(LEGACY_TEXT)
check("夹具：旧族文件已剔除常量与包装",
      "OP_VERSION" not in LEGACY_TEXT and "_VersionedRegistry(registry" not in LEGACY_TEXT)
check("旧族文件（无常量）→ (None, None) 兼容",
      fc.plugin_identity(str(p_legacy)) == (None, None))
p_half = tmp / "server_op_half.py"
p_half.write_text(re.sub(r'^OP_FAMILY.*$', "", FAM_TEXT, flags=re.M))
try:
    fc.plugin_identity(str(p_half))
    check("半残常量（只有 OP_VERSION）应报错", False)
except RuntimeError as e:
    check("半残常量（只有 OP_VERSION）报错并点名", "必须成对" in str(e))

print("\n=== 2) 版本化上传 + 返回实际名 ===")
name1 = fc.ensure_plugin(BASE, "preprocess.anyline", plugin_path=str(p1), tok="t")
check("返回 <基名>@<族版本>", name1 == f"preprocess.anyline@{V1}", name1)
check("落盘名 = <OP_FAMILY>_<ver>.py",
      f"preprocess_g7_{V1}.py" in svc.files, str(sorted(svc.files)))
check("服务端注册的是版本化名（不是基名）",
      f"preprocess.anyline@{V1}" in svc.ops and "preprocess.anyline" not in svc.ops)
u1 = svc.uploads
check("同内容再 ensure：幂等，不上传", fc.ensure_plugin(
    BASE, "preprocess.anyline", plugin_path=str(p1), tok="t") == name1
    and svc.uploads == u1)

print("\n=== 3) F1 核心：改过实现的 v2 与 v1 并存，v1 顶不掉 v2 ===")
v2_text, V2 = sync_common.stamp_family(
    FAM_TEXT.replace('"preprocess.shuffle"', '"preprocess.shuffle"') +
    "\n# F1 测试：等价改一行（换内容 ⇒ 换版本）\n", "preprocess_g7.py")
p2 = tmp / "server_op_v2.py"
p2.write_text(v2_text)
check("v2 版本戳与 v1 不同", V2 != V1, f"{V1} -> {V2}")
name2 = fc.ensure_plugin(BASE, "preprocess.anyline", plugin_path=str(p2), tok="t")
check("v2 返回自己的版本名", name2 == f"preprocess.anyline@{V2}", name2)
check("服务端两个版本【并存】（F1 的根本判据）",
      all(n in svc.ops for n in (name1, name2))
      and svc.ops[name1]["plugin_hash"] != svc.ops[name2]["plugin_hash"])
check("两个版本各自文件都在盘上（内容寻址）",
      f"preprocess_g7_{V1}.py" in svc.files and f"preprocess_g7_{V2}.py" in svc.files)
# 老节点（v1 内容）后跑：必须仍解析到 v1，且不删/不覆盖 v2
again = fc.ensure_plugin(BASE, "preprocess.anyline", plugin_path=str(p1), tok="t")
check("v1 老节点后跑仍拿到 v1（旧版顶不掉新版）", again == name1, again)
check("v2 仍完整存在（未被顶掉）",
      f"preprocess.anyline@{V2}" in svc.ops
      and svc.ops[f"preprocess.anyline@{V2}"]["plugin_file"] == f"preprocess_g7_{V2}.py")
check("v1 后跑未产生新上传（内容早已注册）", svc.uploads == u1 + 1, f"uploads={svc.uploads}")

print("\n=== 4) 旧族文件兼容路径 ===")
# 用真实"三算子族"文件（preprocess.py）造旧形态夹具，基名 preprocess.canny
FAM2_TEXT = (ROOT / "_common" / "server_ops" / "preprocess.py").read_text()
_a = FAM2_TEXT.index("# --- F1 族版本化")
_b = FAM2_TEXT.index("# ---------------------------------------------------------------------------",
                     _a) + len("# ---------------------------------------------------------------------------\n")
LEGACY_CANNY = (FAM2_TEXT[:_a] + FAM2_TEXT[_b:]).replace(
    "    registry = _VersionedRegistry(registry, OP_VERSION)\n", "")
p_legacy2 = tmp / "server_op_legacy_canny.py"
p_legacy2.write_text(LEGACY_CANNY)
nameL = fc.ensure_plugin(BASE, "preprocess.canny", plugin_path=str(p_legacy2), tok="t")
check("旧族文件仍返回基名", nameL == "preprocess.canny", nameL)
check("旧族文件沿用旧落盘名 preprocess_canny.py", "preprocess_canny.py" in svc.files)

print("\n=== 5) call_op / submit_job 自动翻译基名 ===")
sent = {}


def fake_post(base, path, payload, tok=None, timeout=None):
    """替身：只记录提交内容（真 post_json 返回解析后的 dict）。"""
    sent[path] = payload
    return {"job_id": "j1"} if path == "/jobs" else {"outputs": {}}


fc.post_json = fake_post
fc.ensure_plugin(BASE, "preprocess.anyline", plugin_path=str(p1), tok="t")
fc.submit_job(BASE, {"name": "preprocess.anyline", "inputs": {}}, tok="t")
check("submit_job 把基名翻成 v1 实际名",
      sent["/jobs"]["name"] == name1, sent["/jobs"]["name"])
fc.call_op(BASE, "preprocess.anyline", {})
check("call_op 同样翻译", sent["/op"]["name"] == name1, sent["/op"]["name"])
fc.submit_job(BASE, {"name": "sd.sample", "inputs": {}}, tok="t")
check("未登记的基名原样通行（核心/历史算子）",
      sent["/jobs"]["name"] == "sd.sample")

print("\n=== 6) 跨文件冲突：F1 之下结构上不可能，但兜底报错路径要留 ===")
# 6a：同基名、不同内容 ⇒ 版本名不同 ⇒ 服务端【并存】、不产生 409
use_service()          # 切回假服务端（第 5 节换成了记录型替身）
import hashlib
alien = FAM_TEXT.replace('OP_FAMILY = "preprocess_g7"',
                         'OP_FAMILY = "preprocess_other"')
alien, ALIEN_VER = sync_common.stamp_family(alien, "preprocess_other.py")
pa = tmp / "server_op_alien.py"
pa.write_text(alien)
nameA = fc.ensure_plugin(BASE, "preprocess.anyline", plugin_path=str(pa), tok="t")
check("不同内容 ⇒ 不同算子名（结构上不冲突）", nameA == f"preprocess.anyline@{ALIEN_VER}",
      nameA)
check("同名两版仍并存（既不是顶替，也不是 409）",
      all(n in svc.ops for n in (name1, name2, nameA)))

# 6b：兜底路径 —— 服务端确实存在"不同内容的同基名算子"时，客户端必须报错而不是静默顶掉
def stub_409(kind):
    def _get(base, path, tok=None, timeout=None):
        return {"ops": [{"name": "preprocess.anyline", "plugin_hash": "deadbeef" * 8,
                         "plugin_file": "foreign.py"}]}
    def _post(base, path, payload, tok=None, timeout=None):
        raise RuntimeError('POST /admin/plugins -> HTTP 409: {"detail":'
                           '"op conflicts (rolled back)"}')
    return _get, _post


_g, _p = stub_409(409)
fc.get_json, fc.post_json = _g, _p
try:
    fc.ensure_plugin(BASE, "preprocess.anyline", plugin_path=str(p1), tok="t")
    check("服务端已有不同内容同基名算子时必须报错", False, "未报错（= 旧 bug：静默顶掉）")
except RuntimeError as e:
    check("报错并给出 plugins.d 处置指引",
          "409" in str(e) and "删除" in str(e), str(e)[:90])
use_service()

print(f"\nFAILED = {FAIL}")
sys.exit(1 if FAIL else 0)
