from flowx_client import emit, emit_result_preview, param, token
from image_batch import collect, load_batch


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    files, dirpath = collect(param("dir", "~/flowx-input"), param("pattern", "*.png"),
                            param("sort", "name"), param("reverse", "false"),
                            param("limit", 0, cast=int), param("offset", 0, cast=int),
                            param("recursive", "false"))
    ids, names = load_batch(url, files, tok)
    print(f"[image-batch-load] {dirpath} pattern={param('pattern')} "
          f"-> {len(ids)} 张 {names[:3]}{'…' if len(names) > 3 else ''}", flush=True)
    emit(images=ids, count=len(ids), names=names, dir=dirpath)
    if ids:
        emit_result_preview(url, ids[0], tok)


if __name__ == "__main__":
    main()
