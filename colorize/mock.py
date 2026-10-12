"""colorize mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_img = param("image", "mock-image")
print("[colorize][mock] image=%s model=%s size=%s saturation=%s"
      % (_img, param("model", "ddcolor:modelscope"), param("size", "512"),
         param("saturation", "1.0")))
emit(image="mock-colorized-" + str(_img), width=512, height=512,
     chroma=21.4, chroma_in=0.3,
     info='{"variant":"modelscope","dtype":"fp32","size":512,"saturation":1.0}')
