"""flux-detail-refine mock：生成伪重绘图像 id。"""
from flowx_client import emit, param

image = param("image", "mock")
detector = param("detector", "face")
oid = f"mock-flux-refined-{detector}-from-{image}"
print(f"[flux-detail-refine][mock] detector={detector} image={image} -> {oid}")
emit(image=oid, count="1")
