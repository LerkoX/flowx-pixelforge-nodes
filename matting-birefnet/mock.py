"""matting-birefnet mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[matting-birefnet][mock] image=%s variant=%s resolution=%s"
      % (_image, param("variant", "lite"), param("resolution", "1024")))
emit(mask="mock-mask-birefnet-" + str(_image), image="mock-rgba-birefnet-" + str(_image),
     width=512, height=512)
