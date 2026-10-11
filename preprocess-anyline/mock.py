"""preprocess-anyline mock：不调服务，返回伪提示图对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[anyline][mock] image=%s detect_resolution=1280, guassian_sigma=2.0, intensity_threshold=3" % _image)
emit(image="mock-anyline-" + str(_image))
