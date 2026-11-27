"""checkpoint-restore：从命名存档物化新的活对象（免疫对象仓库重启清空）。"""
from flowx_client import emit, param, post_json, token


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    name = param("name").strip()
    if not name:
        raise ValueError("checkpoint-restore: name 不能为空")
    doc = post_json(url, f"/checkpoints/{name}/materialize", {}, tok)
    print(f"[checkpoint-restore] '{name}' -> {doc['kind']} {doc['id'][:8]}"
          f"（{doc['size_bytes'] >> 20}MB）", flush=True)
    emit(object=doc["id"], kind=doc["kind"], size_bytes=doc["size_bytes"])


if __name__ == "__main__":
    main()
