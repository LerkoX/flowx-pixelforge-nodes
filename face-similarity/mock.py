"""face-similarity mock：不调服务，回传伪相似度。"""
from flowx_client import emit, param

sim = 0.87
print(f"[face-similarity][mock] a={param('image_a', 'img_a')} "
      f"b={param('image_b', 'img_b')} -> similarity={sim}")
emit(similarity=sim)
