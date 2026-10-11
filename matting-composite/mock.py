"""matting-composite mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[matting-composite][mock] image=%s bg_mode=%s bg_color=%s"
      % (param("image", "mock-image"), param("bg_mode", "color"), param("bg_color", "#FFFFFF")))
emit(image="mock-composite-" + str(param("image", "mock-image")), width=64, height=64)
