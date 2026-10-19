#!/usr/bin/env python3
"""executor_base.run_op 的 soft 容错出口单测（P1-17，纯 stub 无网络）。

覆盖：
1. soft=True + job 失败 → 返回 {}、输出 soft_failed="true"、打分键置空、
   error 消毒（无 #/反引号/换行——纪律 #10，Studio codec-block 会因此炸解析）；
2. **负例**：soft=False + 同样失败 → RuntimeError 照常抛出（现有行为不变）；
3. soft=True + job 成功 → 走正常路径（emit_keys 正常输出，无 soft_failed）。

运行：python3 nodes/_tools/test-soft-fail.py
"""
import contextlib
import io
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "_common"))

import executor_base as eb  # noqa: E402

FAILED = []
os.environ["SERVICE_URL"] = "http://stub.invalid"
os.environ["SERVICE_TOKEN"] = ""


def check(label, cond):
    print(("✓" if cond else "✗"), label)
    if not cond:
        FAILED.append(label)


def stub_common(monkey_wait):
    eb.ensure_op = lambda url, tok, name: name
    eb.submit_job = lambda *a, **k: "f" * 32
    eb.emit_preview = lambda *a, **k: None
    eb.emit_input_media = lambda *a, **k: None
    eb.wait_job = monkey_wait


def test_soft_fail():
    def boom(*a, **k):
        raise RuntimeError("job xxx failed: 未检测到人脸 #注释\n第二行 ```代码块```")
    stub_common(boom)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = eb.run_op("face.similarity", {"image_a": "a", "image_b": "b"},
                      emit_keys=["similarity"], soft=True)
    out = buf.getvalue()
    check("soft 失败返回空 dict", r == {})
    check("输出 soft_failed=true", 'soft_failed: "true"' in out)
    check("打分键置空", 'similarity: ""' in out)
    check("error 无 # 字符", "#" not in out.split("error:")[-1])
    check("error 无反引号/换行（单行）", "```代码块```" not in out
          and out.count("```") == 2)  # 只剩 flowx-yaml 围栏本身
    check("error 保留了可读原因", "未检测到人脸" in out)


def test_negative_hard_fail():
    def boom(*a, **k):
        raise RuntimeError("job xxx failed: 未检测到人脸")
    stub_common(boom)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            eb.run_op("face.similarity", {"a": 1}, emit_keys=["similarity"],
                      soft=False)
        check("负例：soft=False 必须照常抛出", False)
    except RuntimeError:
        check("负例：soft=False 必须照常抛出", True)


def test_soft_success_normal_path():
    stub_common(lambda *a, **k: {"outputs": {"similarity": {"value": 0.68}}})
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        r = eb.run_op("face.similarity", {"a": 1}, emit_keys=["similarity"],
                      soft=True)
    out = buf.getvalue()
    check("soft 成功走正常路径（返回分值）", r.get("similarity") == 0.68)
    check("成功路径无 soft_failed", "soft_failed" not in out)


def main():
    test_soft_fail()
    test_negative_hard_fail()
    test_soft_success_normal_path()
    print()
    if FAILED:
        print(f"✗ {len(FAILED)} 项失败")
        sys.exit(1)
    print("全部通过")


if __name__ == "__main__":
    main()
