"""instantid-apply：InstantID 身份保持注入（服务端插件，SDXL 专用）。

人脸身份特征经 IPAdapter 机制注入 UNet（weight，官方默认 0.8），
关键点图经 InstantID ControlNet 控制面部位置/姿态（cn_strength，默认 0.8）。
输出 model/control 分别接采样节点的同名端口。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    run_op("instantid.apply", {
        "model": ref(param("model")),
        "ipadapter": ref(param("ipadapter")),
        "controlnet": ref(param("controlnet")),
        "face": ref(param("face")),
        "weight": param("weight", "0.8", cast=float),
        "cn_strength": param("cn_strength", "0.8", cast=float),
        "start_percent": param("start_percent", "0", cast=float),
        "end_percent": param("end_percent", "1", cast=float),
        "cn_start_percent": param("cn_start_percent", "0", cast=float),
        "cn_end_percent": param("cn_end_percent", "1", cast=float),
    }, emit_keys=["model", "control"], check_exists=True)


if __name__ == "__main__":
    main()
