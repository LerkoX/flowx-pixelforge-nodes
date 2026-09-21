"""cond-combine：两段 conditioning 沿序列维拼接（对标 ComfyUI ConditioningCombine），两段提示词同时生效，可突破 77 token 截断；多区域构图 = set_area 产物经本算子串联。暂不支持 SD3 dict 形式 COND"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("cond.combine", {
        "cond_a": ref(param("cond_a")),
        "cond_b": ref(param("cond_b")),
    }, emit_keys=['cond'], timeout=600)


if __name__ == "__main__":
    main()
