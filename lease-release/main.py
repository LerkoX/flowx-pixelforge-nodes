"""lease-release：显式释放本执行的推理服务租约（流水线收尾节点）。

为什么需要它：FIFO 串行化闸门（inference-ensure 1.4.0）按租约排队——流水线
结束后租约要等 lease_idle_release_seconds（默认 300s）空闲超时才自动交还。
把本节点挂在流水线末尾，跑完立即 DELETE /leases/{execution_id} 交还闸门，
下一条排队的流水线下个轮询周期（默认 5s）就能进场。

语义对齐服务端 DELETE /leases/{id}：只摘租约登记、不卸载模型——常驻模型
留给下一条流水线复用（或由其 ensure 清场淘汰）；执行中算子仍受 pin 保护。
幂等：租约不存在/已过期（404）视为成功。
"""
from flowx_client import delete_json, emit, lease_id, param, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    lid = lease_id()
    if not lid:
        print("[lease-release] 无 FLOWX_EXECUTION_ID（非 Studio 环境），无需释放",
              flush=True)
        emit(released="false", lease_id="")
        return
    try:
        delete_json(url, f"/leases/{lid}", tok, timeout=30)
        print(f"[lease-release] 租约 '{lid}' 已释放（闸门交还，排队流水线可进场）",
              flush=True)
        emit(released="true", lease_id=lid)
    except RuntimeError as e:
        if "HTTP 404" in str(e):
            print(f"[lease-release] 租约 '{lid}' 不存在或已过期（视为成功）",
                  flush=True)
            emit(released="false", lease_id=lid)
        else:
            raise


if __name__ == "__main__":
    main()
