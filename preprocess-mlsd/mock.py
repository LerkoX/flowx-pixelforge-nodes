"""preprocess-mlsd mock：不调服务，返回伪提示图对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[mlsd][mock] image=%s detect_resolution=512, thr_v=0.1, thr_d=0.1" % _image)
emit(image="mock-mlsd-" + str(_image))
