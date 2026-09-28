"""face-similarity：两张图目标人脸的身份余弦相似度（服务端插件算子）。

insightface（antelopev2 normed_embedding 点积）量化两图是否同一身份，
InstantID 验收/调试量化用：同人典型 >=0.5，陌生人典型 <0.3。
各自默认取最大脸（face_index=-1），多脸场景可指定按面积降序的第 N 张。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("face.similarity", {
        "image_a": ref(param("image_a")),
        "image_b": ref(param("image_b")),
        "face_index_a": int(param("face_index_a", "-1")),
        "face_index_b": int(param("face_index_b", "-1")),
        "det_thresh": float(param("det_thresh", "0.5")),
    }, emit_keys=["similarity"], check_exists=True)


if __name__ == "__main__":
    main()
