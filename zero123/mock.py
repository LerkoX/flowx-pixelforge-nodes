"""zero123 mock：不访问服务，返回伪对象 id，验证参数注入与输出提取。"""
from flowx_client import emit, param

iid = param("image", "mock-image")
az = param("azimuth", 0.0, cast=float)
el = param("elevation", 0.0, cast=float)
st = param("steps", 50, cast=int)
print(f"[zero123][mock] image={iid} azimuth={az} elevation={el} steps={st}")
emit(image=f"mock-zero123-out-{iid}", width="256", height="256")
