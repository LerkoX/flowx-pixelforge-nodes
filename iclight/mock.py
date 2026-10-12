"""iclight mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_img = param("image", "mock-image")
print("[iclight][mock] image=%s model=%s light_dir=%s steps=%s seed=%s mask=%s bg=%s"
      % (_img, param("model", "iclight:fc"), param("light_dir", "none"),
         param("steps", "25"), param("seed", "12345"), param("mask", "-"),
         param("bg", "-")))
emit(image="mock-relit-" + str(_img), width=768, height=576,
     light_balance=62.0, lum_left=201.3, lum_right=139.3,
     info='{"variant":"fc","light_dir":"none","steps":25,"seed":12345}')
