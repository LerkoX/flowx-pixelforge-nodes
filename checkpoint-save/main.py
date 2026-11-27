"""checkpoint-save：把媒体对象钉到命名存档（服务端 /checkpoints）。

存档独立于对象磁盘缓存开关（始终可用）：显式复制本体 + 元数据到
checkpoints/<name>/，跨重启/跨执行可回滚，不受 TTL/GC/配额影响。
"""
from flowx_client import emit, param, post_json, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    name = param("name").strip()
    media = param("media").strip()
    if not name or not media:
        raise ValueError("checkpoint-save: name 和 media 都不能为空")
    doc = post_json(url, "/checkpoints", {
        "name": name,
        "object_id": media,
        "note": param("note", ""),
        "overwrite": str(param("overwrite", 0)) in ("1", "true"),
    }, tok)
    print(f"[checkpoint-save] '{name}' <- {doc['kind']} {media[:8]}"
          f"（{doc['size_bytes'] >> 20}MB）", flush=True)
    emit(object=media, checkpoint=name, kind=doc["kind"],
         size_bytes=doc["size_bytes"])


if __name__ == "__main__":
    main()
