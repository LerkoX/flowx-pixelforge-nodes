"""embedding-load：Textual Inversion 加载"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("embedding.load", {
        "clip": ref(param("clip")),
        "names": param("names"),
    }, emit_keys=['clip'], timeout=600)


if __name__ == "__main__":
    main()
