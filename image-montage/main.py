from flowx_client import emit_input_media, emit, emit_result_preview, param, token
from image_batch import load_ids
from image_montage import montage, upload_pil


def main():
    url = param("service_url").rstrip("/")
    tok = token()
    ids = load_ids(param("images"))
    # 输入原图物化：取第一张作为对比控件的「原图」侧
    emit_input_media(url, ids[0] if ids else "", tok)
    pil, grid = montage(url, ids, tok, cols=param("cols", 0, cast=int),
                        cell_w=param("cell_w", 0, cast=int),
                        cell_h=param("cell_h", 0, cast=int),
                        pad=param("pad", 10, cast=int),
                        label=param("label", "none"),
                        label_h=param("label_h", 18, cast=int),
                        bg=param("bg", "#161618"), fit=param("fit", "fit"),
                        names=load_ids(param("names", "")))
    print(f"[image-montage] {len(ids)} 张 -> {pil.width}x{pil.height} grid={grid}", flush=True)
    oid = upload_pil(url, pil, tok)
    emit(image=oid, width=pil.width, height=pil.height, grid=grid, count=len(ids))
    emit_result_preview(url, oid, tok)


if __name__ == "__main__":
    main()
