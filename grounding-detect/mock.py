"""grounding-detect mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_img = param("image", "mock-image")
print("[grounding-detect][mock] image=%s prompt=%s sam=%s"
      % (_img, param("prompt", "person"), param("sam_model", "")))
emit(annotated="mock-annotated-" + str(_img), boxes="[[1,2,3,4]]",
     labels='["person"]', scores="[0.93]", count=1,
     mask="mock-mask-" + str(_img), width=512, height=512)
