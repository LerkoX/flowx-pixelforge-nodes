"""echo-node：测试用回声节点（ResetFromNode/continue 多场景验证）。

- text 透传输出（可引用上游，验证重置后数据流按新值重渲染）
- fail=1 注入失败；sleep 模拟耗时
- marker 文件每次执行追加一行（行数 = 执行次数，验证上游不被重跑）
"""
import datetime
import os
import sys
import time

from flowx_client import emit, param


def main():
    text = str(param("text", ""))
    fail = str(param("fail", 0)) in ("1", "1.0", "true")
    sleep = float(param("sleep", 0) or 0)
    marker = str(param("marker", ""))

    if sleep > 0:
        time.sleep(sleep)

    run_at = datetime.datetime.now().strftime("%H:%M:%S.%f")
    print(f"[echo-node] text={text!r} run_at={run_at}", flush=True)

    if marker:
        with open(os.path.expanduser(marker), "a") as f:
            f.write(f"{run_at} {text}\n")

    if fail:
        print("[echo-node] fail=1 注入失败", file=sys.stderr, flush=True)
        sys.exit(1)

    emit(text=text, run_at=run_at)


if __name__ == "__main__":
    main()
