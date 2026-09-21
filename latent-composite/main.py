"""latent-composite：把 src latent 贴入 dst 的 (x,y) 处（图像像素坐标），越界自动裁剪；feather>0（图像像素）边缘线性羽化。带 noise_mask 的 latent 会被拒绝（先 composite 再 set_noise_mask）"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("latent.composite", {
        "dst": ref(param("dst")),
        "src": ref(param("src")),
        "x": param("x", 0, cast=int),
        "y": param("y", 0, cast=int),
        "feather": param("feather", 0, cast=int),
    }, emit_keys=['latent'], timeout=600)


if __name__ == "__main__":
    main()
