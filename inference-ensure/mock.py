"""inference-ensure mock：不调服务，直接输出就绪态（含显存字段）。"""
from flowx_client import emit, param

url = param("service_url", "http://mock-inference:8100")
expect = (param("expect_resident", "") or "").strip()
print(f"[ensure][mock] ready: {url}")
fields = dict(service_url=url, status="ok", gpu_name="mock-gpu", offload_mode="none",
              vram_free_mb=7000, vram_total_mb=8192, vram_headroom_mb=7000,
              resident_models="(empty)")
if expect:
    items = [t.strip() for t in expect.replace(";", ",").split(",") if t.strip()]
    fields.update(expect_resident=",".join(items),
                  expect_resident_hit=f"0/{len(items)}",
                  min_vram_checked="true")
emit(**fields)
