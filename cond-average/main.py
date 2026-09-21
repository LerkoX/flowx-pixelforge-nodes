"""cond-average：两段 conditioning 按权重加权平均（对标 ComfyUI ConditioningAverage）"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("cond.average", {
        "cond_a": ref(param("cond_a")),
        "cond_b": ref(param("cond_b")),
        "weight": param("weight", 0.5, cast=float),
    }, emit_keys=['cond'], timeout=600)


if __name__ == "__main__":
    main()
