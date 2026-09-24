"""instantid-face-analyze mock：不调服务，透传伪 face/kps id。"""
from flowx_client import emit, param

img = param("image", "img")
print(f"[face-analyze][mock] image={img} face_index={param('face_index', '-1')}")
emit(face=f"mock-face-{img}", kps=f"mock-kps-{img}")
