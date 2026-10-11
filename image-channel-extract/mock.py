"""image-channel-extract mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[image-channel-extract][mock] image=%s " % _image + 'channel="luminance"')
emit(image="mock-image-channel-extract-" + str(_image), width=64, height=64)
