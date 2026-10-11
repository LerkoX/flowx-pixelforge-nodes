"""image-hsl mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[image-hsl][mock] image=%s " % _image + 'hue=30, saturation=1.2')
emit(image="mock-image-hsl-" + str(_image), width=64, height=64)
