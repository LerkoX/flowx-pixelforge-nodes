"""face-restore mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[face-restore][mock] image=%s model=%s upscale=%s"
      % (_image, param("model", "gfpgan"), param("upscale", "1.0")))
emit(image="mock-face-restored-" + str(_image), faces=1, width=512, height=512)
