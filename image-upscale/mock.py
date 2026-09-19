"""image-upscale mock：生成伪放大图像 id。"""
from flowx_client import emit, param

image = param("image", "mock")
scale = param("scale", 2.0)
oid = f"mock-upscaled-{scale}x-from-{image}"
print(f"[upscale][mock] image={image} scale={scale} -> {oid}")
emit(image=oid)
