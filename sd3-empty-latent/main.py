"""sd3-empty-latent：SD3.5 空 Latent（16 通道），输出 LATENT 引用。

自包含节点：服务端算子 sd3.latent.empty 由 server_op.py 自注册（ensure_plugin）。
"""
from flowx_client import call_op, emit, ensure_plugin, param, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    w = param("width", 512, int)
    h = param("height", 512, int)
    batch = param("batch_size", 1, int)

    ensure_plugin(url, "sd3.latent.empty", tok=tok)

    out = call_op(url, "sd3.latent.empty",
                  {"width": w, "height": h, "batch_size": batch}, tok, timeout=60)
    print(f"[sd3-latent] {w}x{h} batch={batch} -> latent={out['latent']}",
          flush=True)
    emit(latent=out["latent"], info=out.get("info", ""))


if __name__ == "__main__":
    main()
