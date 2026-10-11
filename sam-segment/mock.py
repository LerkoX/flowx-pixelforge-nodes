"""sam-segment mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_img = param("image", "mock-image")
print("[sam-segment][mock] image=%s points=%s box=%s pick=%s"
      % (_img, param("points", ""), param("box", ""), param("pick", "0")))
emit(mask="mock-mask-sam-" + str(_img), candidates='[{"index":0,"score":0.96,"area":12345}]',
     score=0.96, annotated="mock-annotated-" + str(_img), width=512, height=512)
