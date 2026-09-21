"""cond-set-area：把整图 cond 标记为只在指定区域生效（对标 ComfyUI ConditioningSetArea）"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("cond.set_area", {
        "cond": ref(param("cond")),
        "x": param("x", 0, cast=int),
        "y": param("y", 0, cast=int),
        "width": param("width", 512, cast=int),
        "height": param("height", 512, cast=int),
        "strength": param("strength", 1.0, cast=float),
    }, emit_keys=['cond'], timeout=600)


if __name__ == "__main__":
    main()
