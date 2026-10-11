"""preprocess-scribble mock：不调服务，返回伪提示图对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[scribble][mock] image=%s detect_resolution=512, safe=True" % _image)
emit(image="mock-scribble-" + str(_image))
