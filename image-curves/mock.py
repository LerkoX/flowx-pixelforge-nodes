"""image-curves mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[image-curves][mock] image=%s " % _image + 'channel="all", points="0,0;128,180;255,255"')
emit(image="mock-image-curves-" + str(_image), width=64, height=64)
