"""instantid-apply：InstantID 身份保持注入（服务端插件，SDXL 专用）。

人脸身份特征经 IPAdapter 机制注入 UNet（weight，官方默认 0.8），
关键点图经 InstantID ControlNet 控制面部位置/姿态（cn_strength，默认 0.8）。
输出 model/control 分别接采样节点的同名端口。
去 CN 化：controlnet 留空或 cn_strength=0 → 仅 IPA 身份注入（8GB 卡救命选项）。
"""
from executor_base import run_op
from flowx_client import param, ref


def main():
    cn = param("controlnet", "")
    payload = {
        "model": ref(param("model")),
        "ipadapter": ref(param("ipadapter")),
        "face": ref(param("face")),
        "weight": param("weight", "0.8", cast=float),
        "cn_strength": param("cn_strength", "0.8", cast=float),
        "start_percent": param("start_percent", "0", cast=float),
        "end_percent": param("end_percent", "1", cast=float),
        "cn_start_percent": param("cn_start_percent", "0", cast=float),
        "cn_end_percent": param("cn_end_percent", "1", cast=float),
    }
    # 去 CN 化：留空则不传（None 会撞引擎的对象引用校验；缺省落函数默认值 None）
    if cn:
        payload["controlnet"] = ref(cn)
    run_op("instantid.apply", payload, emit_keys=["model", "control"])


if __name__ == "__main__":
    main()
