"""image-channel-merge mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

r = param("r_image", "mock-r")
print("[image-channel-merge][mock] r=%s g=%s b=%s a=%s" % (r, param("g_image", "mock-g"), param("b_image", "mock-b"), param("alpha_image", "-")))
emit(image="mock-channel-merge-" + str(r), width=64, height=64)
