"""breakout-compose mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[breakout-compose][mock] grid=%s person=%s height_ratio=%s feet_y=%s"
      % (param("grid", "mock-grid"), param("person", "mock-person"),
         param("height_ratio", "0.9"), param("feet_y", "95")))
emit(image="mock-breakout", width=1940, height=1940,
     person_box='{"x": 700, "y": -60, "width": 540, "height": 1746}')
