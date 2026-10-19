"""tripo-sr mock：生成伪网格对象 id。"""
from flowx_client import emit, param

image = param("image", "mock")
fmt = param("format", "glb")
oid = f"mock-mesh-{fmt}-from-{image}"
print(f"[tripo-sr][mock] image={image} -> {oid}")
emit(mesh=oid, num_vertices="1024", num_faces="2048", format=fmt,
     download_url=f"mock://ext/meshes/{oid}")
