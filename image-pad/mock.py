"""image-pad mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

_image = param("image", "mock-image")
print("[image-pad][mock] image=%s L%s R%s T%s B%s fill=%s feather=%s" % (
    _image, param("left", 0), param("right", 0), param("top", 0),
    param("bottom", 0), param("fill", "edge"), param("feather", 32)))
emit(image="mock-image-pad-" + str(_image), mask="mock-mask-pad-" + str(_image),
     width=512, height=512)
