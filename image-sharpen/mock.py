"""image-sharpen mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[image-sharpen][mock] image=%s " % _image + 'mode="usm", percent=200')
emit(image="mock-image-sharpen-" + str(_image), width=64, height=64)
