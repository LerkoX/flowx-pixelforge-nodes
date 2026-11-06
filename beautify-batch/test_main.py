#!/usr/bin/env python3
"""beautify-batch 本机逻辑单测（不依赖推理服务/GPU/numpy）。

用法：python3 beautify-batch/test_main.py
"""
import importlib.util
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("m", HERE / "main.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

FAILED = 0


def check(label, cond):
    global FAILED
    print(("PASS " if cond else "FAIL ") + label)
    if not cond:
        FAILED += 1


# 1) op_kwargs 从环境变量正确组装
os.environ.update({
    "FLOWX_PARAM_CROP_SIZE": "640", "FLOWX_PARAM_WHITEN": "0.6",
    "FLOWX_PARAM_SMOOTH": "0.3", "FLOWX_PARAM_EYE": "0.15",
    "FLOWX_PARAM_SLIM": "0.08", "FLOWX_PARAM_DET_THRESH": "0.25",
})
kw = m.op_kwargs()
check("crop_size 解析", kw["crop"] == 640)
check("whiten 解析", abs(kw["whiten"] - 0.6) < 1e-9)
check("eye/slim 解析", abs(kw["eye"] - 0.15) < 1e-9 and abs(kw["slim"] - 0.08) < 1e-9)
check("det_thresh 解析", abs(kw["det_thresh"] - 0.25) < 1e-9)

# 2) main 全流程（网络层全部打桩）：逐张调算子、收 ID、emit 格式
os.environ.update({
    "FLOWX_PARAM_SERVICE_URL": "http://stub",
    "FLOWX_PARAM_PATTERN": "*.jpg",
    "FLOWX_PARAM_SORT": "name",
    "FLOWX_PARAM_LIMIT": "0", "FLOWX_PARAM_OFFSET": "0",
    "FLOWX_PARAM_REVERSE": "false", "FLOWX_PARAM_RECURSIVE": "false",
})
with tempfile.TemporaryDirectory() as td:
    for n in ("a.jpg", "b.jpg", "c.jpg", "skip.png"):
        pathlib.Path(td, n).write_bytes(b"x")
    os.environ["FLOWX_PARAM_DIR"] = td

    m.ensure_plugin = lambda url, op, plugin_path=None, tok=None: op + "@stubver"
    m.load_batch = lambda url, files, tok: (m.dumps(["id_" + os.path.basename(f) for f in files]),
                                            m.dumps([os.path.basename(f) for f in files]))
    calls = []

    def fake_call_op(url, name, inputs, tok):
        calls.append(inputs)
        return {"image": "out_" + inputs["image"]["$id"]}
    m.call_op = fake_call_op
    m.emit_result_preview = lambda *a, **k: None
    emitted = {}
    m.emit = lambda **fields: emitted.update(fields)

    m.main()

check("只处理 3 张 jpg（png 被 pattern 排除）", len(calls) == 3)
check("逐张调算子", [c["image"]["$id"] for c in calls] == ["id_a.jpg", "id_b.jpg", "id_c.jpg"])
check("算子入参带美化参数", calls[0]["crop"] == 640 and calls[0]["slim"] == 0.08)
check("images 输出为 @JSON 信封", m.loads(emitted["images"]) == ["out_id_a.jpg", "out_id_b.jpg", "out_id_c.jpg"])
check("count=3", emitted["count"] == 3)

print()
if FAILED:
    print(f"{FAILED} 项失败")
    sys.exit(1)
print("全部通过")
