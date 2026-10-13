"""mock：不连服务的假数据（Studio「运行测试」用）。"""


def run(params):
    return {"ok": "true", "action": params.get("action", "status"),
            "detail": "mock: gpu=MockGPU vram_free=8000MB resident=['flux-vae:vae']",
            "resident_count": 1, "lease_count": 0, "vram_free_mb": 8000,
            "policy": "max_resident_models=2,vram_reserve_mb=1024"}
