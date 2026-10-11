"""image-noise mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[image-noise][mock] image=%s " % _image + 'mode="uniform", amount=0.2, seed=3')
emit(image="mock-image-noise-" + str(_image), width=64, height=64)
