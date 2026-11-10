"""cond-concat：多条件并列（复杂编排的核心组合算子）。

与 cond-combine 的区别：
- cond-combine：序列维拼接成一条**更长的提示词**（突破 77 token）
- cond-concat：多条条件**并存**，各自保留区域/权重/时间窗/层区间

序列长不一致时用 repeat 到最小公倍数填充（不改变注意力语义，优于补零）。
"""
import time

from flowx_client import (emit, emit_preview, ensure_plugin, host_base, param,
                          ref, submit_job, wait_job, token)


def main():
    url = param("service_url").rstrip("/")
    pbase = host_base(url)
    tok = token()
    ensure_plugin(url, "cond.concat", tok=tok)

    a = param("cond_a", "")
    b = param("cond_b", "")
    if not a:
        raise RuntimeError("cond-concat: cond_a 必填")
    inputs = {"cond_a": ref(a)}
    if b:
        inputs["cond_b"] = ref(b)

    jid = submit_job(url, {"name": "cond.concat", "inputs": inputs}, tok)
    emit_preview(f"{pbase}/preview/{jid}", 0.0, tok, base=pbase, job_id=jid)
    print(f"[cond-concat] a={a[:8]} b={(b or '-')[:8]}", flush=True)
    result = wait_job(url, jid, tok, timeout=param("job_timeout", 300, int),
                      poll=param("poll_interval", 2, int))
    out = {k: m.get("id", m.get("value"))
           for k, m in result.get("outputs", {}).items()}
    print(f"[cond-concat] -> cond={out['cond']}", flush=True)
    emit(cond=out["cond"])


if __name__ == "__main__":
    main()
