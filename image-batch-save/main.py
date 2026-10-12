from flowx_client import emit, param, token
from image_batch import load_ids, save_batch


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    ids = load_ids(param("images"))
    paths, names, outdir = save_batch(url, ids, param("output_dir", "~/flowx-e2e-out"),
                                      param("filename_prefix", "batch"),
                                      param("start_index", 1, cast=int),
                                      param("zero_pad", 3, cast=int),
                                      param("ext", "png"), tok)
    print(f"[image-batch-save] {len(paths)} 张 -> {outdir}（{names[:2]}"
          f"{'…' if len(names) > 2 else ''}）", flush=True)
    emit(count=len(paths), paths=paths, filenames=names, dir=outdir)


if __name__ == "__main__":
    main()
