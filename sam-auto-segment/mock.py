"""sam-auto-segment mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_img = param("image", "mock-image")
print("[sam-auto-segment][mock] image=%s pps=%s keep=%s"
      % (_img, param("points_per_side", "16"), param("keep", "")))
emit(masks="mock-montage-" + str(_img), selected="mock-selected-" + str(_img),
     selected_overlay="mock-overlay-" + str(_img), mask_ids='["mock-1","mock-2"]',
     count=2, kept=2, sizes='[{"index":0,"area":100,"score":0.99,"bbox":[0,0,9,9],"id":"mock-1"}]',
     union_area=100, bbox='[0,0,9,9]', montage_cols=2, montage_tile=160)
