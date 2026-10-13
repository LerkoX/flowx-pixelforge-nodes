"""image-shadow mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[image-shadow][mock] image=%s mask=%s bg=%s distance=%s angle=%s "
      "blur=%s opacity=%s squash=%s fade=%s color=%s canvas=%s"
      % (param("image", "mock-image"), param("mask", ""), param("background", ""),
         param("distance", "24"), param("angle", "90"), param("blur", "24"),
         param("opacity", "0.45"), param("squash", "0.55"), param("fade", "0.0"),
         param("color", "#000000"), param("canvas", "expand")))
emit(image="mock-shadowed-" + str(param("image", "mock-image")),
     width=512, height=596, shadow_box="80,420,352,176", coverage=0.2,
     info='{"mock": true}')
