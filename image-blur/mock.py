"""image-blur mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[image-blur][mock] image=%s " % _image + 'mode="motion", radius=8, angle=30')
emit(image="mock-image-blur-" + str(_image), width=64, height=64)
