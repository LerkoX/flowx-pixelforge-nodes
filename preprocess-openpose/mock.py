"""preprocess-openpose mock：不调服务，透传伪骨架图 id。"""
from flowx_client import emit, param

iid = f"mock-pose-{param('image', 'img')}"
print(f"[openpose][mock] hand={param('include_hand', 'false')} face={param('include_face', 'false')} -> {iid}")
emit(image=iid)
