"""video-flow-lock mock：不访问服务，验证参数注入与输出形态。"""
from flowx_client import emit, param

_b = param("base", "mock-base-obj")
_r = param("refined", "mock-refined-obj")
print("[video-flow-lock][mock] base=%s refined=%s" % (_b, _r))
emit(video="mock-locked-obj", fps=24, num_frames=141)
