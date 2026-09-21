"""ipadapter-apply：参考图驱动生成（weight×步窗口控制注入）"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("ipadapter.apply", {
        "model": ref(param("model")),
        "ipadapter": ref(param("ipadapter")),
        "clip_vision": ref(param("clip_vision")),
        "image": ref(param("image")),
        "weight": param("weight", 0.8, cast=float),
        "start_percent": param("start_percent", 0.0, cast=float),
        "end_percent": param("end_percent", 1.0, cast=float),
    }, emit_keys=["model"], timeout=600)


if __name__ == "__main__":
    main()
