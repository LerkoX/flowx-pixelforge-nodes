"""tiled-upscale-apply mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_img = param("image", "mock-image")
print("[tiled-upscale-apply][mock] image=%s tile=%s overlap=%s denoise=%s seam=%s"
      % (_img, param("tile", "768"), param("overlap", "96"),
         param("denoise", "0.35"), param("seam_fix", "mask")))
emit(image="mock-upscaled-" + str(_img), pre_upscaled="mock-pre-" + str(_img),
     seam_map="mock-seam-" + str(_img), tiles=4, coverage=1.0, width=1024, height=1024)
