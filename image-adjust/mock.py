"""image-adjust mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[image-adjust][mock] image=%s " % _image + 'brightness=1.4, contrast=1.1')
emit(image="mock-image-adjust-" + str(_image), width=64, height=64)
