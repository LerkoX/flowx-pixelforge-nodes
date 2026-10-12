"""image-watermark mock：不访问服务，验证参数注入与输出提取。"""
from flowx_client import emit, param

print("[image-watermark][mock] image=%s text=%r pos=%s opacity=%s tile=%s logo=%s"
      % (param("image", "mock-image"), param("text", "SAMPLE"), 
         param("position", "bottom-right"), param("opacity", "0.6"),
         param("tile", "false"), param("logo", "")))
emit(image="mock-watermarked-" + str(param("image", "mock-image")),
     width=512, height=512, box="300,440,180,48", box_w=180, box_h=48,
     item_w=180, item_h=48)
