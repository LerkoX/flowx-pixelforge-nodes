"""tiled-upscale-plan mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

_img = param("image", "mock-image")
print("[tiled-upscale-plan][mock] image=%s scale=%s tile=%s overlap=%s"
      % (_img, param("scale", "2.0"), param("tile", "768"), param("overlap", "96")))
emit(plan='{"src":[512,512],"target":[1024,1024],"tile":768,"overlap":96,'
          '"cells":[],"seed":-1,"seam_fix":"mask","pre":"lanczos","denoise":0.35,'
          '"band_denoise":0.25,"steps":20}',
     grid_image="mock-grid-" + str(_img), tiles=4, rows=2, cols=2,
     target_w=1024, target_h=1024, est_samples=80, target_size="1024x1024")
